from pathlib import Path
import os
from .train_methods import DreamBoothDataset, collate_fn, PromptDataset, import_model_class_from_model_name_or_path
from .convert import convert
import warnings
from pathlib import Path


import hashlib
import itertools
import logging
import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from argparse import Namespace
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
import math


logger = get_logger(__name__)

class TrainMultiSubjectSD():

    def setParameters(self, params):
        # params = json.load(args)
        print(params)
        data = {
            # Important
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5" if params["pretrained_model_name_or_path"] is None else params["pretrained_model_name_or_path"],
            "resolution": 512 if params["resolution"] is None else params["resolution"],
            "train_batch_size": 1 if params["train_batch_size"] is None else params["train_batch_size"],
            "max_train_steps": 2000 if params["max_train_steps"] is None else params["max_train_steps"],
            "checkpointing_steps": 1000 if params["checkpointing_steps"] is None else params["checkpointing_steps"],
            "checkpoints_total_limit": params["checkpoints_total_limit"],
            "gradient_accumulation_steps": 1 if params["gradient_accumulation_steps"] is None else params["gradient_accumulation_steps"],
            "trained_model_path": "multi-subject-model" if params["trained_model_path"] is None else params["trained_model_path"],

            # Can be changed but default values set.
            "adam_beta1": 0.9 if params["adam_beta1"] is None else params["adam_beta1"],
            "adam_beta2": 0.999 if params["adam_beta2"] is None else params["adam_beta2"],
            "adam_weight_decay": 1e-2 if params["adam_weight_decay"] is None else params["adam_weight_decay"],
            "adam_epsilon": 1e-08 if params["adam_epsilon"] is None else params["adam_epsilon"],
            "max_grad_norm": 1.0 if params["max_grad_norm"] is None else params["max_grad_norm"],
            "learning_rate": 5e-6 if params["learning_rate"] is None else params["learning_rate"],
            "scale_lr": False if params["scale_lr"] else params["scale_lr"],
            "lr_scheduler": "constant" if params["lr_scheduler"] is None else params["lr_scheduler"],
            "lr_warmup_steps": 500 if params["lr_warmup_steps"] else params["lr_warmup_steps"],
            "lr_num_cycles": 1 if params["lr_num_cycles"] is None else params["lr_num_cycles"],
            "lr_power": 1.0 if params["lr_power"] is None else params["lr_power"],
            "logging_dir": "logs" if params["logging_dir"] is None else params["logging_dir"],
            "report_to": "tensorboard" if params["report_to"] is None else params["report_to"],
            "num_train_epochs": 1 if params["num_train_epochs"] is None else params["num_train_epochs"],
            "with_prior_preservation": False if params["with_prior_preservation"] is None else params["with_prior_preservation"],
            "prior_loss_weight": 1.0 if params["prior_loss_weight"] is None else params["prior_loss_weight"],
            "num_class_images": 100 if params["num_class_images"] is None else params["num_class_images"],
            "center_crop": False if params["center_crop"] is None else params["center_crop"],
            "sample_batch_size": 4 if params["sample_batch_size"] is None else params["sample_batch_size"],
            
            # None
            "allow_tf32": params["allow_tf32"],
            "mixed_precision": params["mixed_precision"],
            "enable_xformers_memory_efficient_attention": params["enable_xformers_memory_efficient_attention"],
            "push_to_hub": params["push_to_hub"],
            "hub_token": params["hub_token"],
            "hub_model_id": params["hub_model_id"],
            "revision": params["revision"],
            "tokenizer_name": params["tokenizer_name"],
            "instance_data_dir": params["instance_data_dir"],
            "class_data_dir": params["class_data_dir"],
            "instance_prompt": params["instance_prompt"],
            "class_prompt": params["class_prompt"],
            "checkpoint_path": params["checkpoint_path"],
            "seed": params["seed"],
            "train_text_encoder": params["train_text_encoder"],
            "resume_from_checkpoint": params["resume_from_checkpoint"],
            "use_8bit_adam": params["use_8bit_adam"],
            "gradient_checkpointing": params["gradient_checkpointing"],
            "prior_generation_precision": params["prior_generation_precision"],
            "local_rank": params["local_rank"],
            "half": params["half"],
            "use_safetensors": params["use_safetensors"]
        }

        if params["instance_data_dir"] is None:
            raise ValueError("Specify `instance_data_dir`")

        if params["instance_prompt"] is None:
            raise ValueError("Specify `instance_prompt`")

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != params["local_rank"]:
            params["local_rank"] = env_local_rank

        if params["with_prior_preservation"]:
            if params["class_data_dir"] is None:
                raise ValueError("You must specify a data directory for class images.")
            if params["class_prompt"] is None:
                raise ValueError("You must specify prompt for class images.")
        else:
            # logger is not available yet
            if params["class_data_dir"] is not None:
                warnings.warn("You need not use class_data_dir without with_prior_preservation.")
            if params["class_prompt"] is not None:
                warnings.warn("You need not use class_prompt without with_prior_preservation.")
        return data


    def train(self, args_dict):
        args = Namespace(**args_dict)
        logging_dir = Path(args.trained_model_path, args.logging_dir)
        print(args)
        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            logging_dir=logging_dir,
            project_config=accelerator_project_config,
        )

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
        # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        # Parse instance and class inputs, and double check that lengths match
        instance_data_dir = args.instance_data_dir.split(",")
        instance_prompt = args.instance_prompt.split(",")
        assert all(
            x == len(instance_data_dir) for x in [len(instance_data_dir), len(instance_prompt)]
        ), "Instance data dir and prompt inputs are not of the same length."

        if args.with_prior_preservation:
            class_data_dir = args.class_data_dir.split(",")
            class_prompt = args.class_prompt.split(",")
            assert all(
                x == len(instance_data_dir)
                for x in [len(instance_data_dir), len(instance_prompt), len(class_data_dir), len(class_prompt)]
            ), "Instance & class data dir or prompt inputs are not of the same length."
        else:
            class_data_dir = args.class_data_dir
            class_prompt = args.class_prompt

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Generate class images if prior preservation is enabled.
        if args.with_prior_preservation:
            for i in range(len(class_data_dir)):
                class_images_dir = Path(class_data_dir[i])
                if not class_images_dir.exists():
                    class_images_dir.mkdir(parents=True)
                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    if args.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif args.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif args.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(class_prompt[i], num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = (
                                class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            )
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.trained_model_path is not None:
                os.makedirs(args.trained_model_path, exist_ok=True)

            if args.push_to_hub:
                repo_id = create_repo(
                    repo_id=args.hub_model_id or Path(args.trained_model_path).name, exist_ok=True, token=args.hub_token
                ).repo_id

        # Load the tokenizer
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
        elif args.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )

        vae.requires_grad_(False)
        if not args.train_text_encoder:
            text_encoder.requires_grad_(False)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if args.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
        )
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=instance_data_dir,
            instance_prompt=instance_prompt,
            class_data_root=class_data_dir if args.with_prior_preservation else None,
            class_prompt=class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(len(instance_data_dir), examples, args.with_prior_preservation),
            num_workers=1,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        # Prepare everything with our `accelerator`.
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        if not  args.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=vars(args))

        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(args.trained_model_path)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.trained_model_path, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.trained_model_path, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)              
                
                if global_step >= args.max_train_steps:
                    break

        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.trained_model_path)

            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.trained_model_path,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()
        if args.checkpoint_path is not None:
            # converting outputs to .ckpt
            print("="*20)
            print("converting outputs to .ckpt")
            convert(args)
            logger.info("Finish training and converting")



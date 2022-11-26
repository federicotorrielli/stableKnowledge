import os

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm


class ImageGenerator:
    def __init__(self, prompt_list: list, powerful_gpu: bool = False, folder_name="output") -> None:
        self.prompt_list = prompt_list
        self._mkdir_if_not_exists(folder_name)
        self.folder_name = folder_name
        # The user needs to be logged-in with huggingface-cli
        self.generator = self._initialize_generator()
        weights = "stabilityai/stable-diffusion-2"
        torch.backends.cudnn.benchmark = True # enabling cuDNN auto-tuner for faster convolution
        if not powerful_gpu:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                weights,
                device_map="auto",
                revision="fp16",
                torch_dtype=torch.float16,
                safety_checker=None)
            # self.pipe.enable_sequential_cpu_offload()
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                weights,
                device_map="auto",
                safety_checker=None)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        self.warmup_pass()

    def warmup_pass(self):
        """
        Warmup pass to initialize the model
        """
        fake_prompt = "A photo of an astronaut riding a horse on mars"
        self.pipe(fake_prompt, num_inference_steps=1)
        print("Warmup pass complete || READY TO GENERATE IMAGES")

    def generate_images(self, steps=30):
        """
        Generates images for each prompt in self.prompt_list
        """
        pbar = tqdm(self.prompt_list)
        i = 1
        for prompt in pbar:
            # Don't generate if the folder already exists
            subfolder_name = prompt.replace(' ', '_').replace(',', '-')
            if not os.path.exists(f"{self.folder_name}/{subfolder_name}/"):
                os.mkdir(f"{self.folder_name}/{subfolder_name}/")
                pbar.set_description(f"Generating: {prompt}")
                image = self.pipe(prompt,
                                  negative_prompt="writing, letters, handwriting, words",
                                  num_inference_steps=steps,
                                  generator=self.generator,
                                  guidance_scale=7.5,
                                  num_images_per_prompt=5)
                for j, img in enumerate(image.images):
                    img.save(f"{self.folder_name}/{subfolder_name}/{i}_{j}.png")
                i += 1

    def _mkdir_if_not_exists(self, param):
        if not os.path.exists(param):
            os.mkdir(param)

    def _initialize_generator(self):
        gen = torch.Generator(device='cuda')
        seed = 26111998
        return gen.manual_seed(seed)

    def set_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list

    def set_folder_name(self, folder_name):
        self.folder_name = folder_name
        self._mkdir_if_not_exists(folder_name)

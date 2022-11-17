import os

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from tqdm import tqdm


# pip install --upgrade diffusers[torch]
# conda install xformers -c xformers/label/dev


class ImageGenerator:
    def __init__(self, prompt_list: list, powerful_gpu: bool = False, folder_name="generated_images") -> None:
        self.prompt_list = prompt_list
        self._mkdir_if_not_exists(folder_name)
        self.folder_name = folder_name
        # The user needs to be logged-in with huggingface-cli
        euler_scheduler = EulerDiscreteScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.generator = self._initialize_generator()
        if not powerful_gpu:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                device_map="auto",
                scheduler=euler_scheduler,
                revision="fp16",
                torch_dtype=torch.float16,
                safety_checker=None).to("cuda")
            # self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                device_map="auto",
                scheduler=euler_scheduler,
                torch_dtype=torch.float32,
                safety_checker=None).to("cuda")
        self.pipe.enable_attention_slicing()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # self.pipe.enable_xformers_memory_efficient_attention()
        self.warmup_pass()

    def warmup_pass(self):
        """
        Warmup pass to initialize the model
        """
        fake_prompt = "A photo of an astronaut riding a horse on mars"
        self.pipe(fake_prompt, num_inference_steps=5)
        print("Warmup pass complete || READY TO GENERATE IMAGES")

    def generate_images(self, steps=30):
        """
        Generates images for each prompt in self.prompt_list
        """
        pbar = tqdm(self.prompt_list)
        i = 0
        for prompt in pbar:
            # Don't generate if the image already exists
            if not os.path.exists(f"{self.folder_name}/{prompt.replace(' ', '_').replace(',', '-')}.png"):
                pbar.set_description(f"Generating: {prompt}")
                image = self.pipe(prompt,
                                  negative_prompt="writing, letters, handwriting, words",
                                  num_inference_steps=steps,
                                  generator=self.generator,
                                  guidance_scale=7.5).images[0]
                image.save(f"{self.folder_name}/{i}_{prompt.replace(' ', '_').replace(',', '-')}.png")
                i += 1

    def generate_one_test_image(self):
        """
        Generates one test image: useful for debugging
        """
        prompt = "An astronaut riding a horse on mars"
        image = self.pipe(prompt, num_inference_steps=30).images[0]
        image.save(f"{self.folder_name}/{prompt.replace(' ', '_').replace(',', '-')}.png")
        print(f"Generated images for prompt: {prompt}")

    def _mkdir_if_not_exists(self, param):
        if not os.path.exists(param):
            os.mkdir(param)

    def _initialize_generator(self):
        gen = torch.Generator(device='cuda')
        seed = 1117437330
        return gen.manual_seed(seed)

    def set_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list

    def set_folder_name(self, folder_name):
        self.folder_name = folder_name
        self._mkdir_if_not_exists(folder_name)


if __name__ == "__main__":
    prompt_list = []
    ig = ImageGenerator(prompt_list, powerful_gpu=True)
    ig.generate_one_test_image()

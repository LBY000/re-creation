# Re-creation Diffusion: Text-to-Image Generation with Harmonized Relationships to Any Layout of Creation

# Introduction
<img src="/pic/castle.png" width="1000px">
In graphic design, the selection of background images is one of the key factors affecting the design effect, but manually filtering suitable background images is often a time-consuming and tedious process. To address this challenge, this paper proposes Re-creation Diffusion, a novel framework without additional training that generates non-conflicting “re-creation” images based on arbitrary layouts. Unlike traditional natural images, “re-creation” images require a harmonious relationship between the layout of creation and the salient objects in the image, i.e., sufficient non-salient space needs to be reserved for the layout of creation. To achieve this goal, we propose a noise-coverage technique based on the existing diffusion model, which effectively solves the space reservation problem by transforming a given layout into an additional noise interference. This result indicate that noise serves as an effective medium for controlling image generation. Meanwhile, we further propose an adaptive cross-attention distribution mechanism to ensure that the generated images achieve optimal results in terms of layout consistency and visual quality. As a new method for graphic design, we also construct an evaluation dataset to validate the effectiveness of our proposed method. The experimental results demonstrate that Re-creation Diffusion significantly outperforms existing methods, generating images that are highly consistent with the given layouts and exhibit superior quality.

# Getting Start

## Installation
```
git clone https://github.com/LBY000/Re-creation.git
cd Re-creation
```



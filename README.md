[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generative-adversarial-transformers/image-generation-on-clevr)](https://paperswithcode.com/sota/image-generation-on-clevr?p=generative-adversarial-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generative-adversarial-transformers/image-generation-on-cityscapes)](https://paperswithcode.com/sota/image-generation-on-cityscapes?p=generative-adversarial-transformers)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generative-adversarial-transformers/image-generation-on-lsun-bedroom-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-bedroom-256-x-256?p=generative-adversarial-transformers)

# GANsformer: Generative Adversarial Transformers
<p align="center">
  <b>Drew A. Hudson* & C. Lawrence Zitnick</b></span>
</p>

*I wish to thank Christopher D. Manning for the fruitful discussions and constructive feedback in developing the Bipartite Transformer, especially when explored within the language representation area, as well as for the kind financial support that allowed this work to happen!

<div align="center">
  <img src="https://cs.stanford.edu/people/dorarad/image1.png" style="float:left" width="340px">
  <img src="https://cs.stanford.edu/people/dorarad/image3.png" style="float:right" width="440px">
</div>

This is an implementation of the [GANsformer](https://arxiv.org/pdf/2103.01209.pdf) model, a novel and efficient type of transformer, explored for the task of image generation. The network employs a _bipartite structure_ that enables long-range interactions across the image, while maintaining computation of linearly efficiency, that can readily scale to high-resolution synthesis. 
The model iteratively propagates information from a set of latent variables to the evolving visual features and vice versa, to support the refinement of each in light of the other and encourage the emergence of compositional representations of objects and scenes. 
In contrast to the classic transformer architecture, it utilizes multiplicative integration that allows flexible region-based modulation, and can thus be seen as a generalization of the successful StyleGAN network. 

**Update: code is now ready for running! Instructions for model training and data prepreation as well as pretrained models will be available soon. Stay Tuned!**  
- [x] Uploading initial code and readme
- [x] Image scampling and visualization script
- [x] Code clean-up and making it ready for running
- [ ] Training and data-prepreations intructions (*in progress*)
- [ ] Pretrained networks (*by March 20*)
- [ ] Further visualizations/animations

## Bibtex
```bibtex
@article{hudson2021gansformer,
  title={Generative Adversarial Transformers},
  author={Hudson, Drew A and Zitnick, C. Lawrence},
  journal={arXiv preprint:2103.01209},
  year={2021}
}
```

## Architecture overview

The GANsformer consists of two networks:
* **Generator**: which produces the images (`x`) given randomly sampled latents (`z`). The latent z has a shape `[batch_size, component_num, latent_dim]`, where `component_num = 1` by default (Vanilla GAN, StyleGAN) but is > 1 for the GANsformer model. We can define the latent components by splitting `z` along the second dimension to obtain `z_1,...,z_k` latent components. The generator likewise consists of two parts:
  * **Mapping network**: converts sampled latents from a normal distribution (`z`) to the intermediate space (`w`). A series of Feed-forward layers. The k latent components either are mapped independently from the `z` space to the `w` space or interact with each other through self-attention (optional flag).
  * **Synthesis network**: the intermediate latents w are used to guide the generation of new images. Images features begin from a small constant/sampled grid of `4x4`, and then go through multiple layers of convolution and up-sampling until reaching the desirable resolution (e.g. `256x256`). After each convolution, the image features are modulated (meaning that their variance and bias are controlled) by the intermediate latent vectors `w`. While in the StyleGAN model there is one global w vectors that controls all the features equally. The GANsformer uses attention so that the k latent components specialize to control different regions in the image to create it cooperatively, and therefore perform better especially in generating images depicting multi-object scenes.
  * **Attention** can be used in several ways
    * **Simplex Attention**: when attention is applied in one direction only from the latents to the image features (**top-down**).
    * **Duplex Attention**: when attention is applied in the two directions: latents to image features (**top-down**) and then image features back to latents (**bottom-up**), so that each representation informs the other iteratively.
    * **Self Attention between latents**: can also be used so to each direct interactions between the latents.
    * **Self Attention between image features** (SAGAN model): prior approaches used attention directly between the image features, but this method does not scale well due to the quadratic number of features which becomes very high for high-resolutions.
     
* **Discriminator**: Receives and image and has to predict whether it is real or fake – originating from the dataset or the generator. The model perform multiple layers of convolution and downsampling on the image, reducing the representation's resolution gradually until making final prediction. Optionally, attention can be incorporated into the discriminator as well where it has multiple (k) aggregator variables, that use attention to adaptively collect information from the image while being processed. We observe small improvements in model performance when attention is used in the discriminator, although note that most of the gain in using attention based on our observations arises from the generator.

## Codebase
This codebase builds on top of and extends the great [StyleGAN2 repository](https://github.com/NVlabs/stylegan2) by Karras et al.  
The GANsformer model can also be seen as a generalization of StyleGAN: while StyleGAN has one global latent vector that control the style of all image features globally, the GANsformer has *k* latent vectors, that cooperate through attention to control regions within the image, and thereby better modeling images of multi-object and compositional scenes.

#### More documentation and instructions will be coming soon!

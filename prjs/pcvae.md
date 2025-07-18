

# **A Theoretical and Practical Guide to Conditional Variational Autoencoders with Pareto Latent Spaces**

**Abstract:** This report provides a comprehensive theoretical and practical guide for constructing and training a Conditional Variational Autoencoder (CVAE) with a Pareto-distributed latent space. Standard VAEs, which rely on a Gaussian prior, often struggle with over-regularization and can fail to effectively model data with heavy-tailed characteristics or rare events. By leveraging the power-law nature of the Pareto distribution, we can design more expressive generative models. We present a full, step-by-step derivation of the CVAE's Evidence Lower Bound (ELBO) objective, including a novel analytical derivation of the Kullback-Leibler (KL) divergence between two Pareto distributions. Furthermore, we detail the application of the reparameterization trick via inverse transform sampling, a critical step for enabling gradient-based optimization. Finally, we discuss advanced topics, including numerical stability, the benefits of learnable priors, and the potential for improved latent space disentanglement afforded by non-Gaussian assumptions. This work serves as a rigorous guide for researchers seeking to move beyond Gaussian-based latent variable models.

## **Section 1: Foundational Framework: The Conditional Variational Autoencoder (CVAE)**

To fully appreciate the novelty and utility of employing a Pareto distribution within a generative model, it is essential to first establish the foundational framework upon which this new architecture is built. This section will briefly trace the evolution from deterministic autoencoders to their probabilistic counterparts, the Variational Autoencoders (VAEs), and finally to the Conditional VAEs (CVAEs) that offer fine-grained control over the data generation process. This progression reveals a move from simple data compression to sophisticated, structured probabilistic modeling, culminating in an objective function that encapsulates a fundamental trade-off between reconstruction accuracy and latent space regularization.

### **1.1 From Autoencoders to Probabilistic Generative Models: The VAE**

The conceptual ancestor of the VAE is the standard autoencoder, a neural network architecture designed for unsupervised learning, primarily for dimensionality reduction and feature learning.1 An autoencoder consists of two components: an encoder, which maps high-dimensional input data to a lower-dimensional representation (the latent code), and a decoder, which attempts to reconstruct the original input from this compressed code. The model is trained by minimizing a reconstruction loss, which measures the discrepancy between the original input and its reconstruction.  
While effective for compression, standard autoencoders learn a deterministic mapping, resulting in a latent space that may not be continuous or structured in a way that is conducive to generation. Sampling a random point from the latent space and passing it through the decoder will likely produce nonsensical output because the decoder has only learned to reconstruct points corresponding to specific encoded inputs.1  
Variational Autoencoders address this limitation by introducing a probabilistic framework.2 Instead of mapping an input  
x to a single point in the latent space, the VAE's encoder, often called the recognition or inference model qϕ​(z∣x), maps x to a probability distribution over the latent space. A latent vector z is then *sampled* from this distribution. The decoder, or generative model pθ​(x∣z), then maps this sampled point z back to a distribution over the original data space, from which the reconstruction is generated.1  
The primary objective in training such a generative model is to maximize the marginal log-likelihood of the data, logp(x), which is given by the integral over all possible latent codes: logp(x)=log∫pθ​(x∣z)p(z)dz. Here, p(z) is a chosen prior distribution over the latent space (typically a standard multivariate Gaussian, N(0,I)). This integral is almost always intractable to compute directly due to the high dimensionality of the latent space and the complexity of the decoder network.3  
To overcome this intractability, VAEs employ variational inference. We introduce the approximate posterior qϕ​(z∣x) and optimize a lower bound on the log-likelihood, known as the Evidence Lower Bound (ELBO).3 The ELBO is derived as follows:  
logp(x)=log∫p(x,z)dz=log∫qϕ​(z∣x)qϕ​(z∣x)pθ​(x,z)​dz  
By Jensen's inequality, since the logarithm is a concave function, we have:  
logp(x)≥∫qϕ​(z∣x)logqϕ​(z∣x)pθ​(x,z)​dz=LVAE​(θ,ϕ;x)  
This lower bound, LVAE​, can be rearranged into a more intuitive form that highlights the core trade-off in the VAE objective 2:  
LVAE​(θ,ϕ;x)=Ez∼qϕ​(z∣x)​\[logpθ​(x∣z)\]−DKL​(qϕ​(z∣x)∣∣p(z))  
The VAE is trained by maximizing this ELBO. The objective consists of two terms:

1. **The Reconstruction Term:** Ez∼qϕ​(z∣x)​\[logpθ​(x∣z)\] encourages the decoder to accurately reconstruct the input x from latent codes z sampled from the encoder's output distribution. This term pushes the model toward high-fidelity data representation.  
2. **The KL Divergence Term:** $D\_{KL}(q\_\\phi(z|x) |

| p(z))$ is the Kullback–Leibler divergence between the approximate posterior qϕ​(z∣x) and the prior p(z). This term acts as a regularizer, penalizing the model if the distribution of encoded data points deviates significantly from the chosen prior. It forces the latent space to be structured and smooth, enabling meaningful generation from randomly sampled latent codes.

### **1.2 Introducing Conditioning for Controllable Generation**

While a standard VAE can generate novel data samples that resemble the training set, it offers no direct control over the specific characteristics of the generated output. For instance, when trained on the MNIST dataset of handwritten digits, a VAE can generate realistic-looking digits, but we cannot ask it to generate a specific digit, say, a "7".7  
Conditional Variational Autoencoders (CVAEs) extend the VAE framework to provide this control by incorporating a conditional variable, c, into the modeling process.7 This condition could be a class label, a descriptive attribute, or even another data modality (e.g., an image to be translated). The condition  
c is provided as an additional input to the encoder, the decoder, and the prior distribution. The model components are thus modified as follows 8:

* **Conditional Encoder (Recognition Model):** qϕ​(z∣x,c)  
* **Conditional Decoder (Generative Model):** pθ​(x∣z,c)  
* **Conditional Prior:** pθ​(z∣c)

The objective of the CVAE is now to maximize the *conditional* log-likelihood, logp(x∣c). Following the same principles of variational inference used for the standard VAE, we derive the conditional ELBO 8:  
logp(x∣c)≥LCVAE​(θ,ϕ;x,c)=Ez∼qϕ​(z∣x,c)​\[logpθ​(x∣z,c)\]−DKL​(qϕ​(z∣x,c)∣∣pθ​(z∣c))  
This objective function is the cornerstone of the CVAE and the central focus of this report. Its structure mirrors that of the standard VAE, but the inclusion of the condition c has profound implications. The reconstruction term now ensures that the model can reconstruct x given both the latent code z and the condition c. The KL divergence term now regularizes the approximate posterior qϕ​(z∣x,c) towards a *conditional* prior pθ​(z∣c).  
This conditional prior is a crucial element. It implies that the "target" distribution for latent codes is no longer a single, fixed distribution for all data points. Instead, it can vary depending on the condition c. For example, when modeling MNIST, the prior for the digit "1" might be centered in one region of the latent space, while the prior for "8" is centered in another.10 The KL divergence term actively enforces this structure, compelling the encoder to learn this class-conditional organization of the latent space. This is a far more sophisticated form of control than simple generation; it is about learning a structured, conditional manifold where different conditions correspond to distinct, well-defined regions within the latent space. The choice of this prior distribution, therefore, directly influences the fundamental trade-off between information preservation (reconstruction) and structural simplicity (regularization), setting the stage for our exploration of a non-Gaussian, heavy-tailed prior.

## **Section 2: The Pareto Distribution as a Latent Prior: Rationale and Properties**

The vast majority of VAE and CVAE implementations rely on a Gaussian distribution for the prior and the approximate posterior. While mathematically convenient due to its closed-form KL divergence and simple reparameterization, the Gaussian assumption imposes a strong inductive bias that may not be suitable for all types of data. This section provides the mathematical definition of the Pareto distribution and builds a theoretical case for its use as a latent prior, arguing that its heavy-tailed nature offers a principled way to address some of the known limitations of Gaussian-based models.

### **2.1 Mathematical Formalism of the Pareto Distribution**

The Pareto distribution is a power-law probability distribution that is often used to describe phenomena where a small number of items account for a large share of the effect (the "80/20 rule") or where events become progressively less common at larger scales.11 The Type I Pareto distribution, which is the focus of this work, is defined by two parameters: a shape parameter  
α and a scale parameter xm​.  
The Probability Density Function (PDF) of a random variable Z following a Pareto distribution is given by 12:  
f(z;α,xm​)=zα+1αxmα​​for z≥xm​  
The parameters are defined as:

* **Shape parameter α\>0:** Also known as the tail index, α controls the "heaviness" or rate of decay of the tail of the distribution. Smaller values of α correspond to a heavier tail, meaning that extreme values are more probable. Conversely, larger values of α lead to a thinner tail where extreme values are rarer.12  
* **Scale parameter xm​\>0:** This parameter defines the minimum possible value of the random variable, establishing the lower bound of the distribution's support, which is \[xm​,∞).12

The Cumulative Distribution Function (CDF), which gives the probability P(Z≤z), is 12:  
F(z;α,xm​)=1−(zxm​​)αfor z≥xm​  
A key characteristic of the Pareto distribution is the conditional existence of its moments. The mean of the distribution is defined only when α\>1, and the variance is defined only when α\>2.14 This property has direct implications for the latent space; for certain values of  
α, the latent variables may not have a finite mean or variance, a stark contrast to the always-finite moments of a Gaussian.

### **2.2 The Theoretical Case for Heavy-Tailed Latent Distributions**

The choice to replace the conventional Gaussian prior with a Pareto distribution is not an arbitrary one. It is a deliberate modeling decision motivated by the desire to overcome well-documented limitations of standard VAEs and to imbue the model with an inductive bias that is better suited for certain types of real-world data.

#### **Critique of the Gaussian Prior**

The standard Gaussian prior, N(0,I), is light-tailed, meaning its probability density falls off exponentially fast. This property can lead to several pathological behaviors during VAE training:

1. **Over-regularization:** The KL divergence term in the ELBO penalizes the approximate posterior for deviating from the prior. With a Gaussian prior, placing an encoded data point far from the origin (the mean of the prior) incurs a quadratically increasing penalty. This can force the model to under-represent outliers or rare but significant features present in the data, as encoding them accurately would result in an prohibitively large KL loss. The model may opt for a poor reconstruction of these points to keep the KL term low, effectively "ignoring" important but infrequent variations.17  
2. **Posterior Collapse:** In cases where the decoder is very powerful (e.g., a large autoregressive model), it may become capable of generating high-quality reconstructions without relying on the information from the latent code z. The optimization will then minimize the KL divergence term by making the approximate posterior qϕ​(z∣x) equal to the prior p(z) for all x. This renders the latent code uninformative, and the VAE fails to learn a useful representation. This phenomenon, known as posterior collapse, is exacerbated by the restrictive nature of the Gaussian prior.18  
3. **The Prior Hole Problem:** The KL term in the ELBO can be rewritten to show that it encourages the *aggregated posterior*, q(z)=Epdata​(x)​\[qϕ​(z∣x)\], to match the prior p(z).20 A fixed, unimodal Gaussian prior forces the aggregated posterior—a potentially complex, multimodal distribution reflecting the true data manifold—into a simple, compact shape. This mismatch can create "holes" in the latent space: regions with high probability under the prior but low probability under the aggregated posterior. Samples drawn from these holes during generation will likely result in unrealistic or nonsensical outputs.20

#### **Power Laws and the Pareto Advantage**

Many phenomena in the natural and social sciences are not well-described by Gaussian distributions but instead exhibit power-law behavior, characterized by heavy tails.13 Examples include the distribution of city populations, financial market returns, word frequencies in text, and the magnitudes of earthquakes.12 If the underlying generative factors of a dataset follow such a distribution, it is logical that a generative model incorporating a similar inductive bias will be more effective.  
The Pareto distribution, being a canonical power-law distribution, offers a compelling alternative to the Gaussian prior.22 Its heavy tail decays polynomially rather than exponentially, allowing it to accommodate outliers and rare events without incurring an excessive KL penalty. This provides the model with greater flexibility to learn an expressive latent representation that more faithfully captures the true data manifold, especially in its extremities.26 By using a prior that is better matched to the potential complexity of the data, we can mitigate the over-regularization and prior hole problems that plague standard VAEs.  
Furthermore, in the context of a CVAE, the parameters of the Pareto prior, p(z|c) \= Pareto(α\_p(c), x\_m(c)), become learnable functions of the condition c. This transforms the prior from a static, fixed assumption into a dynamic, expressive component of the model. A neural network can learn to output the prior parameters α\_p and x\_m based on the input condition c. This allows the model to learn, for example, that one class of data is characterized by frequent, low-magnitude events (requiring a larger α\_p and thus a thinner tail), while another class is defined by rare, high-magnitude events (requiring a smaller α\_p and a heavier tail). This adaptive prior mechanism, where the very shape of the latent space organization changes with the condition, represents a significant increase in modeling power compared to a fixed-prior approach.20

## **Section 3: Derivation of the CVAE Loss Function with a Pareto Latent Space**

This section presents the core mathematical contribution of this report: the complete derivation of the CVAE loss function when the latent space is governed by Pareto distributions. We will address each component of the ELBO—the reconstruction loss and the KL divergence—and provide a rigorous, step-by-step derivation of the analytical formula for the KL divergence between two Pareto distributions under a necessary simplifying assumption.

### **3.1 The Reconstruction Loss Term: Ez∼qϕ​(z∣x,c)​\[logpθ​(x∣z,c)\]**

The first term in the CVAE ELBO is the expected log-likelihood of the data, often referred to as the reconstruction loss. This term quantifies how well the decoder can reconstruct the original input x when given a latent code z (sampled from the encoder's output) and the condition c. The specific mathematical form of this loss is determined by the nature of the data being modeled, not by the distribution of the latent variables.8  
Common choices for the reconstruction loss include:

* **Binary Cross-Entropy (BCE):** For data that is binary or has been normalized to the range $$, such as binarized MNIST images, the decoder's output can be interpreted as the parameters of a Bernoulli distribution for each pixel. The log-likelihood term then becomes the BCE loss, which is standard practice in many VAE implementations.8  
* **Mean Squared Error (MSE):** For real-valued data, such as image pixel intensities normalized to have a specific mean and variance, the decoder's output is often assumed to be the mean of a Gaussian distribution with a fixed variance. In this case, maximizing the log-likelihood is equivalent to minimizing the MSE between the original input and the reconstructed output.

Regardless of its specific form, this expectation is generally intractable to compute directly. Therefore, it is estimated using Monte Carlo integration. In practice, this involves a simple procedure:

1. A latent code z is sampled from the approximate posterior distribution qϕ​(z∣x,c). As will be detailed in Section 4, this sampling must be done using the reparameterization trick to allow for gradient backpropagation.  
2. This sample z, along with the condition c, is passed through the decoder network to produce a reconstruction x^.  
3. The reconstruction loss (e.g., BCE or MSE) is calculated between the original input x and the reconstruction x^.  
   For improved stability, it is common to average this loss over a small number of samples of z (often just one sample is sufficient per training step).8

### **3.2 The KL Divergence Term: $D\_{KL}(q\_\\phi(z|x,c) |**

| p\_\\theta(z|c))$  
The second term in the CVAE ELBO is the KL divergence, which acts as the regularizer. This term measures the "distance" or information loss when approximating the conditional prior distribution pθ​(z∣c) with the encoder's approximate posterior distribution qϕ​(z∣x,c).31 Unlike the reconstruction term, for many pairs of distributions, this term can be calculated analytically, providing a clean, exact value for the loss without the need for sampling.

#### **3.2.1 Problem Formulation and a Critical Assumption**

The general definition of the KL divergence for two continuous probability distributions, q(z) and p(z), defined over a support Z is 32:  
DKL​(q∣∣p)=∫Z​q(z)log(p(z)q(z)​)dz  
In our specific case, we are modeling the latent variables with Pareto distributions. Let us define our two distributions:

* **Approximate Posterior:** qϕ​(z∣x,c)=Pareto(z;αq​,xm,q​). The encoder network, given inputs x and c, outputs the shape parameter αq​ and scale parameter xm,q​.  
* **Conditional Prior:** pθ​(z∣c)=Pareto(z;αp​,xm,p​). The prior network, given input c, outputs the shape parameter αp​ and scale parameter xm,p​.

A direct derivation of the KL divergence for two Pareto distributions with different scale parameters (xm,q​=xm,p​) is highly complex. The resulting integral involves special functions, specifically the Meijer G-function, which are analytically intractable for the purposes of efficient, gradient-based optimization in a deep learning context.34 This intractability presents a significant practical barrier.  
To proceed, we must make a crucial modeling choice that ensures a closed-form solution. **We assume that the scale parameter xm​ is shared and fixed for both the prior and the posterior distributions.** For instance, we can set xm​=1 for both. This simplification is common when dealing with distributions where parameter differences lead to intractable integrals. It also ensures that the support of both distributions, \[xm​,∞), is identical, which is a necessary condition for a finite KL divergence.32 Under this assumption, our distributions become:

* qϕ​(z∣x,c)=Pareto(z;αq​,xm​)  
* pθ​(z∣c)=Pareto(z;αp​,xm​)

Here, the encoder network only needs to output the shape parameter αq​, and the prior network only needs to output αp​. The scale xm​ is a fixed hyperparameter of the model. This is a clear example of how the demands of mathematical tractability can and should inform practical model design.

#### **3.2.2 Intermediate Lemma: Expectation of the Log-Pareto Variate**

Before tackling the full KL divergence integral, it is useful to derive a key intermediate result: the expected value of the logarithm of a Pareto-distributed random variable. Let Z∼Pareto(αq​,xm​). We need to compute Eq​\[log(Z)\].  
By definition of expectation:  
Eq​\[log(Z)\]=∫xm​∞​log(z)⋅f(z;αq​,xm​)dz=∫xm​∞​log(z)⋅zαq​+1αq​xmαq​​​dz  
We can solve this integral using integration by parts, where ∫udv=uv−∫vdu. Let:

* u=log(z)⟹du=z1​dz  
* dv=zαq​+1αq​xmαq​​​dz=αq​xmαq​​z−(αq​+1)dz⟹v=−αq​αq​xmαq​​​z−αq​=−xmαq​​z−αq​

Applying the integration by parts formula:  
Eq​\[log(Z)\]=\[log(z)⋅(−xmαq​​z−αq​)\]xm​∞​−∫xm​∞​(−xmαq​​z−αq​)⋅z1​dz  
First, evaluate the boundary term \[uv\]xm​∞​:

* As z→∞, the term zαq​log(z)​→0 since αq​\>0.  
* At z=xm​, the term is log(xm​)⋅(−xmαq​​xm−αq​​)=−log(xm​).  
  So, the boundary term evaluates to 0−(−log(xm​))=log(xm​).

Next, evaluate the remaining integral:  
∫xm​∞​xmαq​​z−(αq​+1)dz=αq​1​∫xm​∞​αq​xmαq​​z−(αq​+1)dz  
The integral is simply the integral of the Pareto PDF over its entire support, which equals 1\. Thus, the integral term evaluates to αq​1​.  
Combining the parts, we arrive at the lemma:

Eq​\[log(Z)\]=log(xm​)+αq​1​

#### **3.2.3 Step-by-Step Derivation of the KL Divergence**

With the lemma established, we can now derive the analytical formula for the KL divergence between q(z)=Pareto(αq​,xm​) and p(z)=Pareto(αp​,xm​).  
Start with the definition of KL divergence and expand it:  
DKL​(q∣∣p)=∫xm​∞​q(z)\[log(q(z))−log(p(z))\]dz=Eq​\[log(q(Z))−log(p(Z))\]  
Let's write out the log-probability terms:

* log(q(z))=log(zαq​+1αq​xmαq​​​)=log(αq​)+αq​log(xm​)−(αq​+1)log(z)  
* log(p(z))=log(zαp​+1αp​xmαp​​​)=log(αp​)+αp​log(xm​)−(αp​+1)log(z)

Now, find the difference, log(q(z))−log(p(z)):  
log(q(z))−log(p(z))=(log(αq​)−log(αp​))+(αq​−αp​)log(xm​)−((αq​+1)−(αp​+1))log(z)  
\=log(αp​αq​​)+(αq​−αp​)log(xm​)−(αq​−αp​)log(z)  
Now, take the expectation of this difference with respect to q(z):  
DKL​(q∣∣p)=Eq​\[log(αp​αq​​)+(αq​−αp​)log(xm​)−(αq​−αp​)log(Z)\]  
Using the linearity of expectation, we can separate the terms:  
\=log(αp​αq​​)+(αq​−αp​)log(xm​)−(αq​−αp​)Eq​\[log(Z)\]  
Now, substitute the result from our lemma, Eq​\[log(Z)\]=log(xm​)+αq​1​:  
\=log(αp​αq​​)+(αq​−αp​)log(xm​)−(αq​−αp​)(log(xm​)+αq​1​)  
Expand the final term:  
\=log(αp​αq​​)+(αq​−αp​)log(xm​)−(αq​−αp​)log(xm​)−αq​αq​−αp​​  
The terms involving log(xm​) cancel each other out, leaving:  
\=log(αp​αq​​)−αq​αq​−αp​​=log(αp​αq​​)−(1−αq​αp​​)  
This gives us the final, clean analytical formula for the KL divergence between two Pareto distributions with a shared scale parameter xm​:  
DKL​(Pareto(αq​,xm​)∣∣Pareto(αp​,xm​))=log(αq​)−log(αp​)+αq​αp​​−1  
This result is notable for its simplicity and its independence from the scale parameter xm​. It bears a striking structural resemblance to the KL divergence between two Exponential distributions, $D\_{KL}(\\text{Exp}(\\lambda\_q) |  
| \\text{Exp}(\\lambda\_p)) \= \\log(\\lambda\_q) \- \\log(\\lambda\_p) \+ \\lambda\_p/\\lambda\_q \- 1$.36 This is not a coincidence. The Pareto distribution is directly related to the Exponential distribution via a logarithmic transformation: if  
Z∼Pareto(α,xm​), then Y=log(Z/xm​) is exponentially distributed with rate α.11 The KL divergence is known to be invariant under reparameterizations 33, so the structural similarity of these formulas provides a strong theoretical validation of our derivation.  
For ease of reference, the key mathematical formulas required for implementing a CVAE with a Pareto latent space are consolidated in the table below.

| Formula | Mathematical Expression | Role in Model |
| :---- | :---- | :---- |
| Pareto PDF | f(z;α,xm​)=zα+1αxmα​​ | Theoretical basis of the latent distribution. |
| Pareto Inverse CDF | z=U1/αxm​​, where U∼Uniform(0,1) | Reparameterization trick for differentiable sampling from the encoder. |
| E\[log(z)\] for z∼Pareto(α,xm​) | log(xm​)+α1​ | Intermediate step required for the KL divergence derivation. |
| KL Divergence ($q |  |  |
| p$) | DKL​=log(αq​)−log(αp​)+αq​αp​​−1 | Regularization term in the loss function. Assumes a shared scale xm​. |
| Final CVAE Loss (ELBO) | $\\mathcal{L} \= \\mathbb{E}*{U \\sim \\text{Unif}}\\left\[\\log p*\\theta\\left(x \\left | \\frac{x\_m}{U^{1/\\alpha\_q}}, c\\right)\\right.\\right\] \- \\left(\\log(\\alpha\_q) \- \\log(\\alpha\_p) \+ \\frac{\\alpha\_p}{\\alpha\_q} \- 1\\right)$ |

## **Section 4: Practical Implementation: The Reparameterization Trick for the Pareto Distribution**

Having derived the analytical loss function, the next critical step is to ensure that the entire model can be trained end-to-end using gradient-based optimization methods like stochastic gradient descent (SGD) or its variants (e.g., Adam). A key obstacle arises from the sampling operation in the VAE framework, which introduces a stochastic node into the computational graph. This section explains the necessity of the reparameterization trick and provides a concrete derivation for applying it to the Pareto distribution.

### **4.1 The Necessity of Reparameterization in VAEs**

In the VAE forward pass, we must sample a latent vector z from the approximate posterior distribution defined by the encoder: z∼qϕ​(z∣x,c). This sampling process is inherently random and non-differentiable. Consequently, gradients from the reconstruction loss cannot flow "backward" through this stochastic node to update the parameters ϕ of the encoder network.37 This would prevent the encoder from learning how to produce meaningful latent distributions.  
The reparameterization trick elegantly solves this problem by reframing the sampling process.37 Instead of drawing  
z directly from a distribution whose parameters depend on ϕ, we express z as a deterministic and differentiable function of the parameters ϕ and an independent noise variable ϵ drawn from a fixed, parameter-free distribution. That is, we rewrite the sampling as:  
z=gϕ​(ϵ),where ϵ∼p(ϵ)  
Here, gϕ​ is a deterministic function parameterized by ϕ, and p(ϵ) is a simple distribution like a standard uniform or Gaussian. This restructuring moves the source of randomness outside of the main gradient path. The expectation in the reconstruction loss can now be taken with respect to the fixed distribution p(ϵ), allowing the gradient ∇ϕ​ to pass through the deterministic function gϕ​ and update the encoder's parameters:  
∇ϕ​Ez∼qϕ​(z∣x,c)​\[logpθ​(x∣z,c)\]=∇ϕ​Eϵ∼p(ϵ)​\[logpθ​(x∣gϕ​(ϵ),c)\]=Eϵ∼p(ϵ)​\[∇ϕ​logpθ​(x∣gϕ​(ϵ),c)\]  
This reformulation is the conceptual breakthrough that makes VAEs compatible with standard deep learning frameworks that rely on automatic differentiation and backpropagation.37 The reparameterization trick cleanly separates the source of stochasticity from the learnable parameters, enabling end-to-end training.

### **4.2 Deriving the Pareto Reparameterization via Inverse Transform Sampling**

A general and powerful method for deriving the reparameterization function gϕ​ is inverse transform sampling.39 This statistical technique states that if a random variable  
U is drawn from a standard uniform distribution on the interval (0,1), then the random variable X=F−1(U), where F−1 is the inverse of the target cumulative distribution function (CDF), will have the distribution F.40  
We can apply this principle directly to reparameterize the Pareto distribution. Our goal is to sample z from the approximate posterior qϕ​(z∣x,c)=Pareto(z;αq​,xm​). The CDF of this distribution is:  
F(z)=1−(zxm​​)αq​  
To find the inverse CDF, we set u=F(z) for u∈(0,1) and solve for z:

1. Start with the CDF equation:  
   u=1−(zxm​​)αq​  
2. Isolate the term containing z:  
   1−u=(zxm​​)αq​  
3. Take the 1/αq​ power of both sides:  
   (1−u)1/αq​=zxm​​  
4. Solve for z:  
   z=(1−u)1/αq​xm​​

This equation gives us the deterministic function g we need. Now, we must consider the source of randomness. If we let our noise variable ϵ be a sample from a standard uniform distribution, ϵ∼Uniform(0,1), we can set u=ϵ. This gives us the reparameterization z=xm​/(1−ϵ)1/αq​.  
A further simplification is possible. If ϵ is a random variable uniformly distributed on (0,1), then the random variable 1−ϵ is also uniformly distributed on (0,1). Therefore, we can replace (1−ϵ) with another uniform random variable, which we can also call ϵ for simplicity, leading to the more concise form 11:  
z=ϵ1/αq​xm​​,where ϵ∼Uniform(0,1)  
This is the final reparameterization formula for the Pareto distribution. The complete algorithm for generating a differentiable sample z from the approximate posterior qϕ​(z∣x,c) is as follows:

1. Feed the input data x and condition c into the encoder network to obtain the shape parameter αq​.  
2. Draw a single sample ϵ from a standard uniform distribution, ϵ∼Uniform(0,1).  
3. Compute the latent variable z using the deterministic transformation: z=xm​/ϵ1/αq​.

This computed value of z is now a differentiable function of the encoder's output αq​, and thus of the encoder's parameters ϕ. This allows the gradients from the reconstruction loss to flow back through the decoder and this transformation to update the encoder.  
It is crucial to clarify a common point of confusion for implementers: the reparameterization trick is used exclusively for estimating the **reconstruction loss term** of the ELBO. The **KL divergence term**, as derived in Section 3, has a closed-form analytical solution that depends directly on the parameters αq​ and αp​. Therefore, we do not need to sample z to compute the KL divergence; it is calculated directly. The total loss for a training step is the sum of the Monte Carlo-estimated reconstruction loss and the analytically computed KL divergence. This distinction is vital for a correct and efficient implementation.

## **Section 5: Advanced Considerations and Model Enhancements**

Moving from a theoretical derivation to a robust, high-performing research model requires addressing several practical and advanced conceptual issues. This section discusses critical considerations for numerical stability during training, explores the powerful concept of learnable priors as a way to enhance model expressivity, and examines the potential for improved latent space disentanglement afforded by the non-Gaussian nature of the Pareto distribution.

### **5.1 Ensuring Numerical Stability**

When implementing novel architectures, especially those involving non-standard distributions, ensuring numerical stability is paramount to prevent training failures such as exploding gradients or NaN loss values. For the Pareto CVAE, several potential pitfalls must be addressed.

* **Positivity of the Shape Parameter α:** The Pareto shape parameter α is strictly positive (α\>0). However, a standard neural network layer can output any real number. To enforce this constraint on the parameters αq​ (from the encoder) and αp​ (from the prior network), the output layer of these networks must use an appropriate activation function. A common and effective choice is the softplus function, f(x)=log(1+ex), which maps any real number to a positive output. Alternatively, one could have the network output log(α) and then apply an exponential function, though softplus is often preferred for its slightly better numerical properties.  
* **Division by Zero in the KL Term:** The analytical KL divergence formula, DKL​=log(αq​)−log(αp​)+αp​/αq​−1, contains the term αp​/αq​. If the encoder outputs a value of αq​ that is very close to zero, this division can lead to numerical overflow and result in a NaN loss, halting the training process.43 This is a common issue in deep learning when dealing with ratios. A simple and standard solution is to add a small, positive constant (e.g.,  
  ϵ=10−8) to the denominator to prevent it from becoming exactly zero. The term would be implemented as αp​/(αq​+ϵ). This small modification has a negligible impact on the theoretical properties of the loss while dramatically improving training stability.  
* **Support Mismatch and Infinite Divergence:** The KL divergence is formally defined as infinite if the support of the approximate distribution q is not a subset of the support of the true distribution p.32 This means that if there is any event for which  
  q(z)\>0 but p(z)=0, the distributions are considered infinitely different. In our case, the support of a Pareto distribution is $$, is forced to match a prior that is too simplistic.20

A powerful solution to this problem is to make the prior itself learnable.20 Instead of being a fixed distribution, the prior  
pλ​(z) has its own set of parameters λ that are optimized alongside the encoder and decoder parameters. This allows the prior and the aggregated posterior to "meet in the middle," with the prior adapting its shape to better cover the encoded data points, and the aggregated posterior being regularized towards this more flexible target. This often leads to a tighter ELBO, better generative performance, and a mitigation of the prior hole issue.20  
Our proposed CVAE with a Pareto latent space, pθ​(z∣c)=Pareto(αp​(c),xm​), is a simple yet effective instance of a learnable prior. The prior is not fixed; its shape parameter αp​ is determined by a neural network that takes the condition c as input. This allows the model to learn a different prior shape for each class, adapting the latent space structure to the specific characteristics of the data subset defined by c.30  
For researchers seeking to push the boundaries of model expressivity even further, more advanced learnable prior techniques exist. A prominent example is the **VampPrior (Variational Mixture of Posteriors)**.47 The VampPrior models the prior as a mixture of approximate posteriors, each conditioned on a learnable "pseudo-input." The prior takes the form:  
pλ​(z)=K1​k=1∑K​qϕ​(z∣uk​)  
where {uk​}k=1K​ are the learnable pseudo-inputs (which form the parameters λ). This creates a rich, multimodal prior that can approximate the true aggregated posterior far more closely than a simple parametric distribution.47 While more complex to implement, exploring a "VampPrior" composed of a mixture of Pareto distributions represents a logical and promising direction for future research based on the framework developed in this report. The Pareto CVAE can be seen as occupying a compelling "sweet spot" on the spectrum of model complexity: it is significantly more expressive than a standard Gaussian VAE but remains far more computationally tractable and stable than models with highly complex, non-parametric priors like Normalizing Flows 45 or the full VampPrior.

### **5.3 Disentanglement with Non-Gaussian Latents**

One of the most sought-after goals in representation learning is **disentanglement**, the learning of latent representations where each dimension corresponds to a single, semantically meaningful, and independent factor of variation in the data.50 For example, in a dataset of faces, a perfectly disentangled representation might have one latent dimension controlling hair color, another controlling pose, and a third controlling expression, all independently.  
A significant body of research has shown that achieving unsupervised disentanglement with standard VAEs is exceptionally challenging. A fundamental theoretical result demonstrates that for a VAE with a Gaussian latent space, any rotation of the latent coordinates results in another valid latent space that is indistinguishable from the original in terms of its distribution. This rotational symmetry makes it impossible for the model to uniquely identify the "correct" axes corresponding to the true underlying factors of variation, leading to entangled representations.51  
Recent theoretical work has provided a crucial breakthrough: the assumption of **non-Gaussianity** of the latent factors, combined with other mild assumptions like local isometry of the data manifold, is sufficient to provably recover disentangled representations.52 Linear Independent Component Analysis (ICA), a classical method for blind source separation, has long relied on the non-Gaussianity of the source signals for their identifiability. This principle extends to the non-linear setting of VAEs. If the latent prior is non-Gaussian, it is no longer invariant to rotations, which breaks the symmetry that plagues Gaussian models. This provides a strong incentive for the model to align its latent dimensions with the true, statistically independent factors of variation in the data.  
This provides a powerful, higher-order justification for using a Pareto latent space. The choice is not merely about better fitting the data's tail properties (a first-order benefit) or avoiding posterior collapse (a second-order benefit). By imposing a non-Gaussian structure on the latent space, the Pareto CVAE may inherently encourage the learning of more disentangled and thus more interpretable representations.53 This potential for disentanglement as an emergent property of the distributional assumption is one of the most exciting and promising aspects of exploring non-Gaussian VAEs. While early work on disentanglement focused on modifying the ELBO objective (e.g., the  
β-VAE 51), this more fundamental approach suggests that the choice of the prior distribution itself is a critical lever for achieving meaningful representations.

## **Section 6: Synthesis and Recommendations**

This report has provided a comprehensive theoretical and practical framework for developing a Conditional Variational Autoencoder with a Pareto-distributed latent space. By moving beyond the conventional Gaussian assumption, this model offers the potential for greater expressivity, robustness to heavy-tailed data, and improved latent space structure. This concluding section synthesizes the key findings into a unified view of the final loss function, presents a high-level algorithmic outline for implementation, and offers recommendations for future research directions.

### **6.1 The Complete Loss Function: A Unified View**

The ultimate goal of training the Pareto-CVAE is to maximize the Evidence Lower Bound (ELBO), which is equivalent to minimizing its negative. Combining the Monte Carlo-estimated reconstruction term (using the reparameterization trick) and the analytically derived KL divergence term, the complete loss function to be minimized for a single data point (x,c) is:  
Loss=−LCVAE​=LRecon​+LKL​  
where:  
LRecon​=−Eϵ∼Uniform(0,1)​\[logpθ​(x​ϵ1/αq​xm​​,c)\]  
and  
LKL​=DKL​(qϕ​(z∣x,c)∣∣pθ​(z∣c))=log(αq​)−log(αp​)+αq​αp​​−1  
The encoder network outputs αq​=Encoder(x,c), and the prior network outputs αp​=Prior(c). The total loss is then averaged over a batch of data and optimized using gradient descent.

### **6.2 High-Level Algorithmic Outline for Training**

The training procedure for the Pareto-CVAE can be summarized in the following steps:

1. **Initialization:**  
   * Initialize the parameters of the Encoder, Decoder, and Prior neural networks.  
   * Choose a value for the fixed scale parameter xm​ (e.g., xm​=1).  
   * Select an optimizer (e.g., Adam) and a learning rate.  
2. **Training Loop:** For each epoch, iterate through the training data in batches. For each batch of data pairs {(xi​,ci​)}:  
3. Forward Pass:  
   a. For each (xi​,ci​), pass the inputs through the respective networks to obtain the Pareto shape parameters:  
   \* αq,i​=Encoder(xi​,ci​)  
   \* αp,i​=Prior(ci​)  
   b. Generate a random noise sample from a uniform distribution: ϵi​∼Uniform(0,1).  
   c. Apply the reparameterization trick to compute the latent sample: zi​=xm​/ϵi1/αq,i​​.  
   d. Pass the latent sample zi​ and condition ci​ through the decoder to get the reconstruction parameters (e.g., logits for BCE loss): x^i​=Decoder(zi​,ci​).  
4. Calculate Loss:  
   a. Compute the reconstruction loss for each sample, LRecon,i​, using an appropriate function (e.g., Binary Cross-Entropy) between xi​ and x^i​.  
   b. Compute the analytical KL divergence for each sample: LKL,i​=log(αq,i​)−log(αp,i​)+αp,i​/αq,i​−1.  
   c. The total loss for the batch is the average of the sum of these two components over all samples in the batch:  
   Total\_Loss=BatchSize1​∑i​(LRecon,i​+LKL,i​).  
5. **Backward Pass:** Compute the gradients of Total\_Loss with respect to all trainable parameters in the Encoder, Decoder, and Prior networks.  
6. **Optimizer Step:** Update the network weights using the chosen optimizer.  
7. **Repeat:** Continue until convergence.

### **6.3 Concluding Remarks and Future Directions**

This report has systematically detailed the motivation, derivation, and implementation of a Conditional VAE with a Pareto-distributed latent space. We have shown that this choice is not merely an arbitrary substitution but a principled approach to address known shortcomings of Gaussian-based VAEs, such as over-regularization and the failure to model heavy-tailed data. The analytical derivation of the KL divergence and the application of inverse transform sampling for reparameterization provide a clear and tractable path to implementation.  
The potential benefits of this architecture are significant, including more robust modeling of data with outliers or rare events, mitigation of the prior hole problem through a more flexible (and learnable) prior, and the prospect of learning more disentangled and interpretable latent representations due to the non-Gaussianity of the prior.  
This work opens several exciting avenues for future research:

* **Empirical Validation:** A thorough empirical study is needed to validate the hypothesized benefits of the Pareto-CVAE across various datasets, particularly those known to have heavy-tailed characteristics (e.g., in finance, network science, or natural disaster modeling). Performance should be compared against standard Gaussian CVAEs and other heavy-tailed alternatives like Student-t VAEs 17, focusing on reconstruction quality, generative diversity, and quantitative measures of disentanglement.  
* **Flexible Priors:** The model can be extended by incorporating more complex learnable priors. A natural next step would be to implement a prior based on a mixture of Pareto distributions, inspired by the VampPrior framework 47, to capture multimodal aggregated posteriors.  
* **Exploring Other Heavy-Tailed Distributions:** The framework presented here can be adapted to other heavy-tailed distributions like the Weibull or Lomax distributions, which may offer different inductive biases suitable for other types of data.54  
* **Generalized Pareto Distribution:** For even greater flexibility, the model could be extended to use the Generalized Pareto Distribution (GPD), which includes the Pareto Type I, exponential, and other distributions as special cases, though this would likely require a more complex derivation of the loss terms.11

In conclusion, the Pareto-CVAE represents a valuable and theoretically grounded step away from the ubiquitous Gaussian assumption in generative modeling, offering a promising tool for researchers seeking to build more powerful and expressive models of complex, real-world data.

#### **Works cited**

1. What is a Variational Autoencoder? \- IBM, accessed July 17, 2025, [https://www.ibm.com/think/topics/variational-autoencoder](https://www.ibm.com/think/topics/variational-autoencoder)  
2. How to Sample From Latent Space With Variational Autoencoder | HackerNoon, accessed July 17, 2025, [https://hackernoon.com/how-to-sample-from-latent-space-with-variational-autoencoder](https://hackernoon.com/how-to-sample-from-latent-space-with-variational-autoencoder)  
3. Variance reduction properties of the reparameterization trick \- Proceedings of Machine Learning Research, accessed July 17, 2025, [http://proceedings.mlr.press/v89/xu19a/xu19a.pdf](http://proceedings.mlr.press/v89/xu19a/xu19a.pdf)  
4. CS598LAZ \- Variational Autoencoders, accessed July 17, 2025, [https://slazebni.cs.illinois.edu/spring17/lec12\_vae.pdf](https://slazebni.cs.illinois.edu/spring17/lec12_vae.pdf)  
5. Evidence, KL-divergence, and ELBO \- Massimiliano Patacchiola, accessed July 17, 2025, [https://mpatacchiola.github.io/blog/2021/01/25/intro-variational-inference.html](https://mpatacchiola.github.io/blog/2021/01/25/intro-variational-inference.html)  
6. Kullback-Leibler Divergence \- Medium, accessed July 17, 2025, [https://medium.com/@amit25173/kullback-leibler-divergence-4566a3b0892f](https://medium.com/@amit25173/kullback-leibler-divergence-4566a3b0892f)  
7. Exploring Advanced Generative AI | Conditional VAEs \- Analytics Vidhya, accessed July 17, 2025, [https://www.analyticsvidhya.com/blog/2023/09/generative-ai-conditional-vaes/](https://www.analyticsvidhya.com/blog/2023/09/generative-ai-conditional-vaes/)  
8. Conditional Variational Auto-encoder — Pyro Tutorials 1.9.1 ..., accessed July 17, 2025, [https://pyro.ai/examples/cvae.html](https://pyro.ai/examples/cvae.html)  
9. Conditional\_VAE/Conditional\_VAE.ipynb at master · nnormandin/Conditional\_VAE \- GitHub, accessed July 17, 2025, [https://github.com/nnormandin/Conditional\_VAE/blob/master/Conditional\_VAE.ipynb](https://github.com/nnormandin/Conditional_VAE/blob/master/Conditional_VAE.ipynb)  
10. Conditional Variational Autoencoder for Prediction and Feature Recovery Applied to Intrusion Detection in IoT, accessed July 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5621014/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5621014/)  
11. Pareto distribution \- Wikipedia, accessed July 17, 2025, [https://en.wikipedia.org/wiki/Pareto\_distribution](https://en.wikipedia.org/wiki/Pareto_distribution)  
12. Pareto Distribution \- GeeksforGeeks, accessed July 17, 2025, [https://www.geeksforgeeks.org/pareto-distribution/](https://www.geeksforgeeks.org/pareto-distribution/)  
13. Pareto Distribution Definition \- Statistics How To, accessed July 17, 2025, [https://www.statisticshowto.com/pareto-distribution/](https://www.statisticshowto.com/pareto-distribution/)  
14. Probability Playground: The Pareto Distribution, accessed July 17, 2025, [https://www.acsu.buffalo.edu/\~adamcunn/probability/pareto.html](https://www.acsu.buffalo.edu/~adamcunn/probability/pareto.html)  
15. Understanding the Pareto Distribution: A Comprehensive Guide \- DataCamp, accessed July 17, 2025, [https://www.datacamp.com/tutorial/pareto-distribution](https://www.datacamp.com/tutorial/pareto-distribution)  
16. Pareto distribution — Probability Distribution Explorer documentation, accessed July 17, 2025, [https://distribution-explorer.github.io/continuous/pareto.html](https://distribution-explorer.github.io/continuous/pareto.html)  
17. $t^3$-Variational Autoencoder: Learning Heavy-tailed Data with ..., accessed July 17, 2025, [https://openreview.net/forum?id=RzNlECeoOB](https://openreview.net/forum?id=RzNlECeoOB)  
18. \[D\] Variational Autoencoders are not autoencoders : r/MachineLearning \- Reddit, accessed July 17, 2025, [https://www.reddit.com/r/MachineLearning/comments/al0lvl/d\_variational\_autoencoders\_are\_not\_autoencoders/](https://www.reddit.com/r/MachineLearning/comments/al0lvl/d_variational_autoencoders_are_not_autoencoders/)  
19. Failures of Variational Autoencoders and their Effects on Downstream Tasks, accessed July 17, 2025, [http://www.gatsby.ucl.ac.uk/\~balaji/udl2020/accepted-papers/UDL2020-paper-056.pdf](http://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-056.pdf)  
20. 7\_VAE\_priors \- Jakub M. Tomczak, accessed July 17, 2025, [https://jmtomczak.github.io/blog/7/7\_priors.html](https://jmtomczak.github.io/blog/7/7_priors.html)  
21. A Contrastive Learning Approach for Training Variational Autoencoder Priors, accessed July 17, 2025, [https://proceedings.neurips.cc/paper/2021/hash/0496604c1d80f66fbeb963c12e570a26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0496604c1d80f66fbeb963c12e570a26-Abstract.html)  
22. Power law \- Wikipedia, accessed July 17, 2025, [https://en.wikipedia.org/wiki/Power\_law](https://en.wikipedia.org/wiki/Power_law)  
23. The Power of the Power Law. Most of the output comes from little of… | by Nir Zicherman | Medium, accessed July 17, 2025, [https://medium.com/@NirZicherman/the-power-of-the-power-law-b24181fe1b92](https://medium.com/@NirZicherman/the-power-of-the-power-law-b24181fe1b92)  
24. Understanding the Power Law Distribution \- Enqurious, accessed July 17, 2025, [https://www.enqurious.com/blog/understanding-the-power-law-distribution](https://www.enqurious.com/blog/understanding-the-power-law-distribution)  
25. Recognizing Power-law Graphs by Machine Learning Algorithms using a Reduced Set of Structural Features, accessed July 17, 2025, [https://www.inf.ufpr.br/amlima/eniac2019.pdf](https://www.inf.ufpr.br/amlima/eniac2019.pdf)  
26. Variational Autoencoder with Learned Latent Structure, accessed July 17, 2025, [http://proceedings.mlr.press/v130/connor21a/connor21a.pdf](http://proceedings.mlr.press/v130/connor21a/connor21a.pdf)  
27. Heavy-Tailed Diffusion Models \- arXiv, accessed July 17, 2025, [https://arxiv.org/html/2410.14171v1](https://arxiv.org/html/2410.14171v1)  
28. Heavy-Tailed Diffusion Models \- OpenReview, accessed July 17, 2025, [https://openreview.net/forum?id=tozlOEN4qp](https://openreview.net/forum?id=tozlOEN4qp)  
29. Variational AutoEncoders with Student-t distribution for large portfolios and IV curves \- Probability & Partners, accessed July 17, 2025, [https://probability.nl/wp-content/uploads/2022/11/The\_use\_of\_Variational\_AutoEncoders\_in\_a\_financial\_context.pdf](https://probability.nl/wp-content/uploads/2022/11/The_use_of_Variational_AutoEncoders_in_a_financial_context.pdf)  
30. A variational autoencoder trained with priors from canonical pathways increases the interpretability of transcriptome data, accessed July 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11251626/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11251626/)  
31. KL Divergence for Machine Learning \- Dibya Ghosh, accessed July 17, 2025, [https://dibyaghosh.com/blog/probability/kldivergence.html](https://dibyaghosh.com/blog/probability/kldivergence.html)  
32. 2.4.8 Kullback-Leibler Divergence, accessed July 17, 2025, [https://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf](https://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf)  
33. Kullback–Leibler divergence \- Wikipedia, accessed July 17, 2025, [https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler\_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)  
34. Lomax distributions \- Kullback Leibler divergence \- Cross Validated \- Stats Stackexchange, accessed July 17, 2025, [https://stats.stackexchange.com/questions/34353/lomax-distributions-kullback-leibler-divergence](https://stats.stackexchange.com/questions/34353/lomax-distributions-kullback-leibler-divergence)  
35. How to calculate Kullback-Leibner divergence when both distribution P and Q contain zero-probable elements? \- Reddit, accessed July 17, 2025, [https://www.reddit.com/r/MachineLearning/comments/2wb8y0/how\_to\_calculate\_kullbackleibner\_divergence\_when/](https://www.reddit.com/r/MachineLearning/comments/2wb8y0/how_to_calculate_kullbackleibner_divergence_when/)  
36. Kullback-Leibler divergence of two exponential distributions with different scale parameters, accessed July 17, 2025, [https://math.stackexchange.com/questions/2589976/kullback-leibler-divergence-of-two-exponential-distributions-with-different-scal](https://math.stackexchange.com/questions/2589976/kullback-leibler-divergence-of-two-exponential-distributions-with-different-scal)  
37. Reparameterization trick \- Wikipedia, accessed July 17, 2025, [https://en.wikipedia.org/wiki/Reparameterization\_trick](https://en.wikipedia.org/wiki/Reparameterization_trick)  
38. REINFORCE vs Reparameterization Trick \- Syed Ashar Javed, accessed July 17, 2025, [https://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/](https://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/)  
39. Machine Learning Trick of the Day (4): Reparameterisation Tricks \- The Spectator, accessed July 17, 2025, [https://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/](https://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/)  
40. Chapter 5 Simulation | Data Science and Statistical Computing \- Louis Aslett, accessed July 17, 2025, [https://www.louisaslett.com/Courses/DSSC/notes/simulation.html](https://www.louisaslett.com/Courses/DSSC/notes/simulation.html)  
41. Generating Normal Random Variables \- Part 1: Inverse Transform Sampling \- T-Tested, accessed July 17, 2025, [https://www.ttested.com/generating-normal-random-variables-part-1/](https://www.ttested.com/generating-normal-random-variables-part-1/)  
42. The Pareto Distribution \- Random Services, accessed July 17, 2025, [https://www.randomservices.org/random/special/Pareto.html](https://www.randomservices.org/random/special/Pareto.html)  
43. Modern PyTorch Techniques for VAEs: A Comprehensive Tutorial, accessed July 17, 2025, [https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/](https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/)  
44. Understanding KL Divergence in PyTorch \- GeeksforGeeks, accessed July 17, 2025, [https://www.geeksforgeeks.org/deep-learning/understanding-kl-divergence-in-pytorch/](https://www.geeksforgeeks.org/deep-learning/understanding-kl-divergence-in-pytorch/)  
45. FlowPrior: Learning Expressive Priors for Latent Variable Sentence Models \- ACL Anthology, accessed July 17, 2025, [https://aclanthology.org/2021.naacl-main.259/](https://aclanthology.org/2021.naacl-main.259/)  
46. PriorVAE: encoding spatial priors with variational autoencoders for small-area estimation | Journal of The Royal Society Interface, accessed July 17, 2025, [https://royalsocietypublishing.org/doi/10.1098/rsif.2022.0094](https://royalsocietypublishing.org/doi/10.1098/rsif.2022.0094)  
47. VAE with a VampPrior \- Proceedings of Machine Learning Research, accessed July 17, 2025, [http://proceedings.mlr.press/v84/tomczak18a/tomczak18a.pdf](http://proceedings.mlr.press/v84/tomczak18a/tomczak18a.pdf)  
48. VAE with a VampPrior \- Proceedings of Machine Learning Research, accessed July 17, 2025, [https://proceedings.mlr.press/v84/tomczak18a.html](https://proceedings.mlr.press/v84/tomczak18a.html)  
49. Variational Autoencoder with a Normalizing Flow prior \- Pyro, accessed July 17, 2025, [https://pyro.ai/examples/vae\_flow\_prior.html](https://pyro.ai/examples/vae_flow_prior.html)  
50. Towards Latent Space Disentanglement of Variational AutoEncoders for Language \- uu .diva, accessed July 17, 2025, [https://uu.diva-portal.org/smash/get/diva2:1682344/FULLTEXT01.pdf](https://uu.diva-portal.org/smash/get/diva2:1682344/FULLTEXT01.pdf)  
51. Disentangling Disentanglement in Variational Autoencoders \- Proceedings of Machine Learning Research, accessed July 17, 2025, [http://proceedings.mlr.press/v97/mathieu19a/mathieu19a.pdf](http://proceedings.mlr.press/v97/mathieu19a/mathieu19a.pdf)  
52. When is Unsupervised Disentanglement Possible?, accessed July 17, 2025, [https://proceedings.neurips.cc/paper/2021/file/29586cb449c90e249f1f09a0a4ee245a-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/29586cb449c90e249f1f09a0a4ee245a-Paper.pdf)  
53. The Influence of KL Regularization on Disentanglement, accessed July 17, 2025, [https://apxml.com/courses/vae-representation-learning/chapter-5-disentangled-representation-learning-vaes/kl-regularization-disentanglement](https://apxml.com/courses/vae-representation-learning/chapter-5-disentangled-representation-learning-vaes/kl-regularization-disentanglement)  
54. Kullback–Leibler divergence and the Pareto–Exponential ..., accessed July 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4864786/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4864786/)  
55. Entropy-based goodness-of-fit test: application to the Pareto distribution \- AIP Publishing, accessed July 17, 2025, [https://pubs.aip.org/aip/acp/article-pdf/1553/1/155/11454199/155\_1\_online.pdf](https://pubs.aip.org/aip/acp/article-pdf/1553/1/155/11454199/155_1_online.pdf)
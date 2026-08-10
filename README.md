# Cosmic birefringence from a joint analysis of ACT and Planck

This codebase performs a joint analysis of cosmic birefringence in ACT and Planck, reproducing the results of Eskilt 2026.

The code is divided into three parts/folders:
* ```spectrum_pipeline```: This creates the power spectra using NaMaster or pspy from the released maps of ACT and Planck. It also creates the LCDM spectra which get beam-convolved. Additionally, it saves the coupling kernels between masks and the MCM matrix for the covariance calculations.
* ```likelihood```: This bins the spectra and calculate the covariance elements. It also performs corrections. These are packaged into likelihood files in the .npz fileformat by Numpy.
* ```beta_mcmc```: This runs the cosmic birefringence estimation using the likelihood .npz files. This is based on the Minami&Komatsu method and priors on miscalibration angles.

# Pre-requisite
Before running the code, you need to download all the necessary files and add the path to the variables in ```tools/data_loading.py``` and ```data/masks/make_npipe_act_window.py```.
* **ACT**: Most of the files are on their [website](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/). You need the ```act_dr6.02_std_AA_night``` srcfree and standard time split maps from [here.](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_maps_standard_get.html) You also need the beam files [here](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_harmonic_beams_profiles_get.html), and the ACT masks [here.](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_pspipe_window_functions_get.html) Additionally, you need some files I could not find on the website but which you can find on NERSC or Globus. You need the k-space filter products which includes the point-source mask ```source_mask_15mJy_and_dust_rad5.fits``` and the filter matrices kspace_tf for ACT x ACT and ACT x Planck.
* **Planck**: You need the NPIPE A/B data split maps and beams. They are available on NERSC or [here.](https://pla.esac.esa.int/pla/#home)
* **Masks**: The ACT masks you got from the step above, but you need the Planck mask as well. The nearly-full sky mask and 30% Galactic cut mask can be found [here.](https://drive.google.com/file/d/1DBjOoidstPnaNkjzRJ6HGlmboD2aenvn/view) The point-source mask needed for the ACT x Planck spectra can be found [here.](https://drive.google.com/file/d/1XWm92GcV4iygYO5uY8zVwcGRAs9XHnsr/view?usp=sharing)


# How Do I Run It?
1. We start by making the mask used for ACT x Planck which is the ACT mask + point-sources found in Planck. This is done in ```data/masks/make_npipe_act_window.py```. Set up the path to the Planck point-source mask you downloaded from the step above and run ```python make_npipe_act_window.py```.
2. You then go to the ```spectrum_pipeline``` folder. You need to set up the ```pipeline_config.py```. Right now, it is set up for the full joint analysis, but you might want to test with fewer bands to make things run faster. First you run ```python run_pipeline.py``` to create the power spectra and then ```python run_coupling.py``` to run the coupling kernels for the masks.
3. Then we make the likelihood files. The ```likelihood/likelihood_config.py``` is also set up for baseline result, but make the changes needed for the likelihood file you want to create. Run ```python run_likelihood.py```. This create a .npz file.
4. Then we can actually measure cosmic birefringence! We go to ```beta_mcmc``` folder, and we set up the ```config.py``` file there. When we are happy, we start the run ```python run_beta_mcmc.py```. The run should output chain file, trace file and a corner plot file.
 
# Need help?
If you struggle with anything, feel free to ask me for help! Please send me an email here: j.r.eskilt@astro.uio.no. You can also open a Github issue.

# Acknowledgment of AI-assisted tools
This codebase was developed with the help of Anthropic's Claude large language models. It aided the writing, debugging and commenting of the code. All scientific decisions were my own, and I verified the code myself.

## Citation
Feel free to use the code as you see fit, but if you use it for published results, please cite
* J. R. Eskilt, arXiv preprint (2026), arXiv:2608.06480 [astro-ph.CO]

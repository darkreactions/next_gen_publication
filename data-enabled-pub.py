# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %% [markdown]
# <center><h1>A novel dimethylammonium lead iodide perovskite</h1></center>
# <p>
#
# <center><h3>Zhi Li<sup>1</sup>, Mansoor Ani Najeeb<sup>2</sup>, Venkateswaran Shekar<sup>2</sup>, Matthias Zeller<sup>3</sup>, Emory M. Chan<sup>1</sup>, Alexander J. Norquist<sup>2</sup>, Joshua Schrier<sup>4</sup></h3></center>
# <br>
#
# 1. Molecular Foundry, Lawrence Berkeley National Laboratory, 1 Cyclotron Road, Berkeley, California 94720, USA
# 2. Department of Chemistry, Haverford College, 370 Lancaster Avenue, Haverford, Pennsylvania 19041, USA
# 3. Department of Chemistry, Purdue University, West Lafayette, Indiana 47907, USA
# 4. Department of Chemistry, Fordham University, 441 E. Fordham Road, The Bronx, New York, 10458, USA
# %% [markdown]
# # Abstract
#
# Organic-inorganic metal halide perovskites are a diverse family of materials with excellent optoelectronic properties that lend themselves to next-generation photovoltaic devices.  Accurate structure determination of new compounds by single crystal X-ray diffraction requires high quality single crystals.  Using an automated, high-throughput approach for perovskite single crystal discovery (RAPID) based on Inverse Temperature Crystallization (ITC), we explored synthesis conditions for the formation of high quality single crystals of dimethylammonium lead iodide. Using 198 experiments, we identified and characterized a new polymorph of dimethylammonium lead iodide. The comprehensive experimental data capture provided by the ESCALATE (Experiment Specification, Capture and Laboratory Automation Technology) software allows us to observe trends in compound formation.  The approach presented here is designed to be generalizable to different synthetic routes for the acceleration for materials discovery.
#
#
# **Keywords:** Metal halide perovskites, High-throughput, inverse temperature crystallization
# %% [markdown]
# # Introduction
# The excellent optoelectronic properties of metal halide perovskites have driven intense interest in these materials, resulting in great improvements in solution processable photovoltaic devices and a power conversion efficiency of 23.7%[[1](https://www.nrel.gov/pv/cell-efficiency.html)]. The composition and structure of metal halide perovskites determine electronic properties such as carrier mobility and the electronic band structure[[2](https://doi.org/10.1021/cm702405c)]. However, access to suitable crystalline perovskites for detailed property and structural characterization has been hindered by limited understanding of underlying processes through which large high quality single crystals can be grown.
#
# Many diverse synthetic routes exist for the growth of metal halide perovskite single crystals, including antisolvent vapor-assisted crystallization,[[3](https://iopscience.iop.org/article/10.1088/2053-1591/aab8b4/meta)] seeded solution crystal growth,[[4](https://doi.org/10.1126/science.aaa5760)]  slow evaporation,[[5](https://doi.org/10.1038/ncomms8338)] and inverse temperature crystallization (ITC)[[6](https://doi.org/10.1038/ncomms8586)]. ITC is a promising choice for rapid discovery of new perovskite materials because it can be used to grow high quality crystals without the need for long growth times (<3 hours). ITC relies upon and the intriguing phenomenon that the solubility of the product crystal _decreases_ with increased temperature (retrograde solubility), and the microscopic mechanism of this process is an area of active research.  Initial reports  of metal halide perovskite formation by ITC were limited to methylammonium lead iodide (MAPbI<sub>3</sub>), methylammonium lead bromide (MAPbBr<sub>3</sub>), and formamidinium lead iodide (FAPbI<sub>3</sub>)[[7](https://doi.org/10.1038/srep11654)]. Expanding the library of known perovskite crystal structures is slow, because successful crystal growth via ITC requires the simultaneous optimization of a large number of interdependent parameters such as reagent concentration, reaction temperature, and the pH of the solutions. Recently, we described a platform for robot-accelerated perovskite investigation and discovery (RAPID), which we demonstrate using the inverse temperature crystallization of metal halide perovskite single crystals. Using RAPID, we discovered ITC routes to the crystallization of three additional perovskites, including the production and characterization of a novel compound Acetamedinium lead iodide (AcetPbI<sub>3</sub>). Comprehensive data capture is provided by the ESCALATE (Experiment Specification, Capture and Laboratory Automation Technology) software, allowing us to retroactively study reaction outcomes to determine underlying formation mechanisms.
#
# In this paper, we used RAPID to explore the formation of halide perovskites containing [dimethylammonium](http://www.chemspider.com/Chemical-Structure.2284514.html)  using the ITC growth process.  Previous work has only been able to produce dimethylammonium lead iodide as thin films and powders, so the atomic structure is unknown.  We produced large high-quality single crystals, which we use to determine the first reported crystal structure for this compound.  As a demonstration of the comprehensive data capture enabled by ESCALATE, we present a next-generation publication that includes detailed interactive visualization and exploration of the complete set of experimental data.
# %% [markdown]
# # Results and Discussion
#
# ## Reaction outcomes
#
# Reaction outcomes are plotted as a function of the reaction composition, using the scoring scheme described in the <a href="#methodSynthesis">Methods</a> section.  The highest quality crystal outcomes (score = 4) are represented as red points. Lower quality outcomes are depicted as yellow, green, and blue for scores of 3, 2, 1, respectively.)
#
#
# Each point in the following figure represents an experiment and its color indicates the crystal score. The plot can be rotated by clicking and dragging the mouse. Experiment details can be obtained by clicking on any point.
#
# The convex hull shows the experimental space defined by the model set of stock solutions; individual points can be outside of this space if they came from experiments whose stock solutions where the actual prepared stock solutions were slightly more or less concentrated than the nominal description.
#
# Temperature variations as a function of well position are known in experiments using 96-well plates.  An infrared camera was used to determines the actual temperature of each well.  The deviations between the set-point and the actual temperature for each well are largest at the corners and edges of the 96-well plate.  The corrected temperature values can be included in the dataset for each individual reaction. A thermal image of each reaction tray is shown below.
#
# **DEVELOPMENT IN PROGRESS**
#
# **Note:** Be sure to execute all cells in the notebook to enable this to work.
#
# **Warning:** Experiments are clickable only when the convex hull is turned off. To do so, toggle the "Show Convex Hull" button.
#
# **Note:** Clicking on "Show X-Y axes" will show a projection into the organic/inorganic plane.
#
# **Note:** Clicking on the legend will show/hide the correspoding crystal scores in the plot.

# %%
# Run this cell to expand the notebook to take up the entire width of the window
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ng_pub_backend import JsMolFigure
from IPython.display import Image
import plotly.graph_objs as go
import pandas as pd
import importlib
import ng_pub_backend
import sys
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# %%
# Setting up data sources
sys.path.append(
    '/home/jupyter/sd2e-community/perovskite-data/data_connected_pub/')

importlib.reload(ng_pub_backend)
data_path = '/home/jupyter/sd2e-community/versioned-dataframes/perovskite/perovskitedata/0032.perovskitedata.csv'
base_path = '/home/jupyter/sd2e-community/perovskite-data/data_connected_pub'


# %%
# plotting data
f = ng_pub_backend.Figure1(data_path, base_path=base_path)
f.plot

# %% [markdown]
# ## Infrared and powder crystal diffraction characterization
# %% [markdown]
#  <a href="#methodIR">Infrared spectrsocopy was performed</a> on the samples to verify the inclusion of the amine.  This is evident in the N-H stretching band centered at 3042 cm<sup>-1</sup>

# %%
# plot infrared spectra
data = pd.read_csv(base_path+'/files/IR.CSV', header=None)
x = data.iloc[:, 0]
y = data.iloc[:, 1]
fig2 = ng_pub_backend.XYPlot(x, y)
fig2.plot(r'$\text{Wavenumbers } cm^{-1}$', 'Transmittance %', reversed=True)

# %% [markdown]
# <a href="#methodPowderXRD">Powder X-Ray diffraction was performed</a> to probe the three-dimensional structure and phase purity of the sample.  The powder diffraction pattern did not match the similuated pattern for any reported phases, suggesting that these crystals exhiibit an unreported structure.
#
# **SHOWN BELOW:** Our measured spectra
#
# **IN DEVELOPMENT:** ability to compare to reference spectra

# %%
# plot powder XRD spectrum

data = pd.read_csv(base_path+'/files/powder_pattern.txt',
                   header=None, skiprows=3, delimiter=' ')
x = data.iloc[:, 0]
y = data.iloc[:, 1]
annotations = [(11.74, 470, '0 1 0'), (15.93, 50, '0 1 1'), (20.30, 50, '1 1 0'),
               (25.91, 245, '0 2 1'), (29.88, 90, '1 1 2'), (31.23, 130, '1 2 0'),
               (32.78, 110, '0 2 2')]
ref_data = pd.read_csv(base_path+'/files/POWDER Cu.txt',
                       skiprows=4, delim_whitespace=True)
trace = go.Bar(x=ref_data['2th.(Ka)'],
               y=[-30 for i in range(len(ref_data['Int.(Ka)']))], width=0.2)
fig3 = ng_pub_backend.XYPlot(x, y)
fig3.plot(r'$2\theta \text{ (degree)}$', 'Intensity (a.u)',
          annotations=annotations, extra_traces=[trace])

# %% [markdown]
# ## Crystal Growth
#
# Using optimized crystal growth conditions, elucidated using automated experiments, large single crystals were grown in bench scale experiments:

# %%
# display single crystal sample
Image(filename=base_path+'/files/Dimethyl.png')

# %% [markdown]
# ### Javascript based 3D/interactive visualization of the solved structure
# <a href="#methodXRD">Single crystal X-Ray diffraction data were collected</a> to determine the structure of the unknown phase.  The structure of the new phase [AcetH][PbI3], is shown below.  Other polymorphs of this compound have been reported previously.  Their respective structures are shown below as well.

# %%
cif_paths = ['MAN_112_2_2_b_0m_RT_Final SHORT.cif', 'files/HUTVAV.search1.cif',
             'files/HUTVAV01.search1.cif', 'files/HUTVAV03.cif']
fig_names = ['New Crystal', 'HUTVAV', 'HUTVAV01', 'HUTVAV03']
doi_values = {'HUTVAV': '10.5517/cc1j04rf',
              'HUTVAV01': '10.5517/ccdc.csd.cc1j88bd', 'HUTVAV03': '10.5517/ccdc.csd.cc1m81lj'}
#cif_paths = ['MAN_112_2_2_b_0m_RT_Final SHORT.cif']
#fig_names = ['New Crystal']

fig4 = JsMolFigure(cif_paths, fig_names, doi_values, widget_side=600)
fig4.plot

# %% [markdown]
# # Machine Learning
# Synthetic chemistry datasets often exclude failed results (“dark reactions”) and suffer from anthropogenic bias in experiment choices, limiting machine learning (ML) models trained  on  such  data.  In  contrast, RAPID’s combination  of  high-throughput  experimentation, randomized  reaction  parameters  and  complete  data  capture is  ideal  for  training  and  evaluating machine  learning  models.   As  a  demonstration,  we  exported ESCALATE's default  set  of  82 reaction conditions (e.g., concentrations, temperature, stir rate) and organic property descriptors (e.g., molecular weight, atoms number, functional groups) and constructed a variety of machine learning models using the Scikit-Learn Python library.   Binary   classifier   models were   constructed to   distinguish   between experiments resulting in high quality single crystals (Class 4) and non-Class 4 outcomes.  The good  performance  of  the  1-  NN  approach is indicative of an interpolation (rather than extrapolation) problem, which in turn is indicative of the high quality of the full, quasi-random-sampled dataset. Figures display learning curves for Dimethylammonium perovskite using the accuracy score.

# %%
importlib.reload(ng_pub_backend)

ml_section = ng_pub_backend.MLSection(data_path=data_path)

# Defining models
knn_model = KNeighborsClassifier(n_neighbors=1, weights="uniform")
ml_section.add_model(knn_model)

knn_model2 = KNeighborsClassifier(n_neighbors=3, weights="uniform")
ml_section.add_model(knn_model2)

svm_model = SVC(gamma='auto')
ml_section.add_model(svm_model)

# Define learning rates
learning_rate = [0.02, 0.06, 0.1, 0.33, 0.55, 0.78]

ml_section.run_models(learn_rate=learning_rate, n_splits=5)


# %%
# Plot metrics. Options are 'accuracy', 'recall', 'F1' and 'precision'
ml_section.plot(metric='accuracy')

# %% [markdown]
# # Conclusions
#
# In this work, we describe report the first crystal structure for dimethylammonium lead iodide.  This is important because, ... Our next steps are ...
#
# An ongoing data dashboard of experiments and machine learning performance can be found at (http://escalation.sd2e.org/dashboard)
#
# %% [markdown]
# # Materials and Methods
# <a id="method"></a>
#
# ## Materials
# <a id="methodMaterials"></a>
# [Lead iodide](https://www.sigmaaldrich.com/catalog/product/aldrich/211168?lang=en&region=US) and [formic acid](https://www.sigmaaldrich.com/catalog/product/sigald/f0507?lang=en&region=US) were purchased from Sigma Aldrich Chemicals with 99% and >=95% purity. [Dimethylammonium iodide](https://www.greatcellsolar.com/shop/dimethylammonium-iodide.html) (98%) was purchased from Greatcell Solar. γ-Butyrolactone (GBL) (>=98%) was purchased from Spectrum Chemical. All chemicals were used without additional purification. Stock solutions of XXX and ... were prepared and used for the high-throughput synthesis, along with neat GBL and formic acid. ... link to data...
#
#
#
# ## High-throughput synthesis
# <a id="methodSynthesis"></a>
# Experiments were performed on a [Hamilton Microlab Ⓡ NIMBUS4](https://www.hamiltoncompany.com/automated-liquid-handling/platforms/microlab-nimbus) with four independent pipetting channels for solution transfer. The liquid handler is placed inside a controlled atmosphere inside the fume hood. Stock solutions were stored in polypropylene containers and used only once to avoid solution contamination. Pipette tips were  replaced between different types of solutions. Inverse temperature crystallization (ITC) was carried out on Hamilton Heater and Shaker (HHS), which can be heated to a nominal maximum of 105 ℃ (actual maximum temperature measured by infrared thermometry is 95 ℃) and a maximum vortexing speeds of 2000 rpm.  Prepared stock solutions are placed at designated locations on NIMBUS operation deck. Reactions were perofrmed in 8x43cm (diameter x height) glass vials were used as reaction vessels, with 96 vials  loaded on an SBS formatted aluminum heating block (Symyx Technologies) on the HHS. The vials were pre-heated to set temperature before addition of GBL, inorganic/organic and organic-only stock solutions. The first addition of formic acid was followed by  10-20 minutes of shaking to dissolve precipitation, and then a subsequent formic acid addition.   Then the solutions are kept at 95 ºC for crystallization for 2.5 hours. During crystallization, NIMBUS was kept undisturbed and fume hood ventilation was minimized by closing fume hood sash. Ambient environmental conditions such as fume hood temperature, humidity etc were recorded throughout the time of the reaction and was later used for data analysis and optimization.[DATA REF] All time, temperature, and reagent dispense instructions are generated as a machine readable file by ESCALATE[x], with quasirandom sampling of the composition space.  At completion, crystal size and quality for each sample is scored, following the scheme of our previous work[x] and work done by Cooper and coworkers:[13] reaction outcomes are scored into four classes: (1) clear solution without any solid; (2) amorphous powder; (3) polycrystalline material (crystal size < 0.1 mm); and (4) large (>0.1 mm) single crystal(s) suitable for structure determination by single crystal x-ray diffraction. Crystal scores, synthesis parameters, and metadata were uploaded to ESCALATE for later use.
#
# ## Infrared Spectroscopy
# <a id="methodIR"></a>
# Infrared spectroscopy data were collected on an FTIR instrument with an ATR data sample attachment, from 4000 to 500 cm<sup>-1</sup>.
#
#
# ## Powder X-Ray Diffraction
# <a id="methodPowderXRD"></a>
# Powder X-ray diffraction data were collected on a Rigaku Miniflex powder X-ray diffractometer, using a CuKa source (wavelength of 1.54Å).
#
#
#
# ## Single Crystal X-ray Diffraction
# <a id="methodXRD"></a>
# Data was collected using a Rigaku Rapid II R-axis curved image plate diffractometer with a Mo-Ka X-ray microsource (l = 0.71073 Å) with a laterally graded multilayer (Goebel) mirror for monochromatization. A single crystal was mounted on a Mitegen micromesh mount using a trace of mineral oil and cooled in-situ to 100(2) K for data collection. Frames were collected using the dtrek option of CrystalClear-SM Expert 2.1 b32,[i] reflections were indexed and processed with HKL3000,[ii] and the files scaled and corrected for absorption using Scalepack. Frames were collected, reflections were indexed and processed, and the files scaled and corrected for absorption using APEX3 v2018.1-0.[iii] For both structures, the heavy atom positions were determined using SHELXS-97.[iv] All other non-hydrogen sites were located from Fourier difference maps. All non-hydrogen sites were refined using anisotropic thermal parameters with full matrix least squares procedures on F o 2 with I > 2s(I). Hydrogen atoms were placed in geometrically idealized positions. All calculations were performed using SHELXL-2018/3.[v] Relevant crystallographic data are listed in Table XXX.
# %% [markdown]
# # Acknowledgments
#
# This study is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001118C0036. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of DARPA.  Work at the Molecular Foundry was supported by the Office of Science, Office of Basic Energy Sciences, of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231. JS acknowledges the Henry Dreyfus Teacher-Scholar Award (TH-14-010).

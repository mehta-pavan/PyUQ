Components of the c-APK methodology

Pavan Mehta

1. Introduction

This report introduces the main components of the c-APK methodology. The
structure of the report is as follows:

-  Section 2: A brief introduction of c-ANOVA, Krigging and POD is
   outlined.
-  Section 3: User guide – The strategies involved using the supplied
   Python script along with this report.
-  Section 4: Numerical investigations – Test case: Lid driven cavity.

2. Literature Review

2.1 Anchored ANOVA Decomposition

The ANOVA decomposition technique (Yang et al., 2012; Margheri & Sagaut,
2016)⁠ or High Dimensional Model Representation (Rabitz and Alış, 1999)
is a functional decomposition of a high dimensional function in to
series of subsequent lower dimensional functional evaluation
`(1) <#anchor>`__ within a domain Ω. The first term on the right hand
side or the zeroth order term, *f*\ :sub:`0` , is a constant throughout
the domain. The second or first order term
*f*\ :sub:`i`\ *(x*\ :sub:`i`\ *)*\ describes the independent action of
*x*\ :sub:`i`\ in the entire domain; any non-linearities of the
functions are addressed too (Rabitz and Alış, 1999)⁠.

== ======== ===
\  |image0| (1)
== ======== ===

The second order term or the third term in `(1) <#anchor>`__ describes
the joint effect of *x*\ :sub:`i`\ and *x*\ :sub:`j`, this argument
follows for all higher order terms. A special point to note about the
last term *f*\ :sub:`i\ j...n`\ *(x*\ :sub:`i\ j..n`\ *)*, it
contributes towards any residue arising from all terms considered
together. For the integral function *f(x)*, can be evaluated either by
Dirac or Lebesgue measure. The advantage of the former over the later:
functional values can be evaluated at discrete points as per any
sampling strategy or quadrature technique. In this work, the Dirac
measure methodology is introduced. Popularly known as Anchored ANOVA
decomposition or cut-HDMR in the literatures. For notational convenience
we closely, follow Yang et al. (2012)⁠ and Margheri and Sagaut (2016)⁠.

The zeroth order term is evaluated at an “Anchor” point. An anchor point
is a fixed point in space usually denoted as *c*\ :sub:`1,`\ *,
c*\ :sub:`2`\ *, c*\ :sub:`3`\ *, …, c*\ :sub:`N` in an *N* –
dimensional space. These anchor points are generally taken at the center
of computational domain. However, this choice is arbitrary and accuracy
varies as per the choice of these anchor points (Zhang et al., 2011,
2012)⁠. The first and second order terms are computed as per equations
`(2) <#anchor-3>`__ and `(3) <#anchor-4>`__ respectively. Here,
*q*\ :sup:`i`\ :sub:`1` and *q*\ :sup:`j`\ :sub:`2` are the quadrature
points in first and second dimensions respectively. The higher order
terms can be computed in a similar fashion. From `(3) <#anchor-4>`__, it
can be clearly seen; for computation of the second order terms, it
requires first order term. Hence, computation of higher order terms can
be expensive. Moreover, it is found that the majority of the variance is
captured by the first and second order terms. Thereby, truncation of
`(1) <#anchor>`__ after second order terms does not produce a larger
error. In `(4) <#anchor-5>`__ *Є*\ :sub:`T` is the truncation error.

== ======== ===
\  |image1| (2)
== ======== ===

== ======== ===
\  |image2| (3)
== ======== ===

== ======== ===
\  |image3| (4)
== ======== ===

2.2 Proper Orthogonal Decomposition

Proper Orthogonal Decomposition has two other sister methods,
“Karhuen-Loeve Expansion” and the “Principal Component Analysis” . The
fundamental question is to find the pairs of orthogonal basis function
where the data could be projected onto this new sub-space. This
sub-space may not necessarily have the same dimensions as the original
data itself. In Principal Component Analysis, this new set of orthogonal
basis (also refereed to as principal components) are such that, the
first principal component has the maximum variance, followed the second
component which not only is orthogonal to the first but has next maximum
variance of the data and so on. In order to numerically, evaluate the
components we introduce the “method of snapshots”.

The Snapshot matrix, *S* is given by `(5) <#anchor-6>`__; where *f(x)*
is an *N –* dimensional function and are refereed to as snapshots for
total number of *p* functional evaluation.Here, *I*\ :sub:`p` is the
identity matrix of shape, *p x p* and *1*\ :sub:`p`\ is the matrix of
ones of the same shape. Next, we compute the Singular value
decomposition of this snapshot matrix `(6) <#anchor-7>`__. Each column
of the matrix *U* is a principal component; there is no advantage in
keeping all the *p* components. Hence, we truncate it to a finite order
*r* and the orthogonal basis vectors *Ψ* are given by
`(7) <#anchor-8>`__

== ======== ===
\  |image4| (5)
== ======== ===

== ======== ===
\  |image5| (6)
== ======== ===

== ======== ===
\  |image6| (7)
== ======== ===

In Karhuen – Loeve expansion the solution is projected over an
orthogonal basis and expanded as per `(8) <#anchor-9>`__. The main issue
was in addressing the computation of these orthogonal vectors *Ψ* which
is as per the earlier discussion.

== ======== ===
\  |image7| (8)
== ======== ===

Furthermore, the coefficients *β* in `(9) <#anchor-10>`__ are computed
using krigging of the coefficients *ν*\ :sup:`s`\ for a solution at any
non-sampled location. However, for the purposes of this study and c-APK
algorithm, equation `(9) <#anchor-10>`__ is not used. POD is only used
for smoothing noisy Quantity of Interest using `(8) <#anchor-9>`__. For
computing the solution for any non-sampled location, Krigging is
directly applied, detailed in the next section.

== ======== ===
\  |image8| (9)
== ======== ===

Finally, to find the coefficients *ν*\ :sup:`s`\ :sub:`m`\ in
`(8) <#anchor-9>`__, the correlation matrix, *C*\ is computed from the
snapshot matrix, *S* as per `(10) <#anchor-12>`__. The coefficient
*ν*\ :sup:`s`\ :sub:`m`\ is the s-th component of the m-th right
eigenvector.

== ======== ====
\  |image9| (10)
== ======== ====

2.3 Krigging

Krigging or Gaussian Process Regression evolves from multivariate
Gaussian theory. The term “krigging” is coined after the South African
mining engineer, Dane Krige. The definition for a Gaussian process as
stated in Soto et al, (2011)⁠; “\ *A Gaussian Process is a collection of
random variables, any finite number of which have (consistent) joint
Gaussian distributions*.”. Formerly written in `(11) <#anchor-14>`__,
where the process is fully defined by its mean (*μ*) and the covariance
matrix (*Σ*). It is to be noted that a Gaussian distribution is over
vectors while, while a Gaussian process is over functions.

== ========= ====
\  |image10| (11)
== ========= ====

The covariance matrix is used to generate functional values at any
non-sampled location. In `(12) <#anchor-16>`__, *k,*\ is a covariance
function also refereed to as “\ *Kernel*\ ” in Machine Learning
terminology. It is solely depended on the corresponding distances
between two sample points, implying that the function's output value is
heavily depended on its neighboring samples. It is obvious now, that
relative accuracy in interpolation is better than extrapolation. Also,
the results may vary with the choice of Kernel. Hence, a good Kernel
approximation is required. Loosely speaking, the choice of the Kernel
can be based on the type of fitting required. For example, if its in our
prior knowledge that the system behaves in a quadratic sense, a choice
of Kernel resembling the quadratic nature would be wiser to use. We
introduce two Kernel functions here, the “Square Exponential” as given
in Soto et al., (2011)⁠ and “Polynomial Cubic Spline” kernel as per
Margheri and Sagaut, (2016)⁠ in `(13) <#anchor-17>`__ and
`(14) <#anchor-18>`__ respectively. In `(14) <#anchor-18>`__ *θ* is
known as hyper-parameter and its values is taken to be 0.1.

== ========= ====
\  |image11| (12)
== ========= ====

== ========= ====
\  |image12| (13)
== ========= ====

== ========= ====
\  |image13| (14)
== ========= ====

3. User Guide

This section introduces the strategies involved in using the Python
scripts. The scripts of ANOVA decomposition, Krigging and POD are
discussed before getting into the `Software
Architecture <#anchor-22>`__. Primarily because, these scripts form the
basis of the c-APK method and can be used individually too. At first we
introduce index mapping for tracking the data.

3.1 Index Mapping

For tracking data corresponding to a particular solution or a value we
introduce a technique of “index mapping”. From, `Figure
1 <#anchor-23>`__, two sets in a form of array is used, one for the real
data and the other for the indice’s. The need for an index array arises
from its usage in calling the a particular solution automatically, given
that python indexing system uses integers only.

.. figure:: Pictures/10000000000001F8000001CAB5AF2F8C0F6BFDA8.png
   :alt: 
   Figure 1: Index Mapping
   :width: 3.5744in
   :height: 3.248in

   Figure 1: Index Mapping

The shape of both the array is (*data sets x N*), where *N* is the total
number of dimensions. For example, consider the dimension of the problem
is 4 and the number of quadrature points = 2 per dimension. Moreover,
let the number of data sets be all possible combinations of these
quadrature points. Hence, the total number of data sets = 4^2 = 16.
Specifically for anova decomposition or otherwise to maintain
generality, we start incrementing the last index first, followed by the
second last index and on on. This discussion can be summarized visually
in `Figure 2 <#anchor-24>`__. Here, the quadrature points in dimensions
1, 2, 3, 4 are [0.25, 0.75], [-0.25, -0.75], [0.25, 0.75], [-0.25,
-0.75] respectively. The row numbers are an interesting quantity, by
calling the particular row number the corresponding solution to the
quadrature points can be called.

.. figure:: Pictures/10000000000002C00000027321015D2CC47D6F74.png
   :alt: 
   Figure 2: Index mapping example
   :width: 5.3409in
   :height: 4.7563in

   Figure 2: Index mapping example

3.2 ANOVA Decomposition

The python script “anova.py” performs the anchored ANOVA decomposition
in 4 dimension with truncation dimension = 2. The important variables
along with usage is commented in the file itself. Here, the usage of the
script, data input and output for this file is illustrated. As an input,
the mean, 1\ :sup:`st` order and 2\ :sup:`nd` order quadrature terms are
provided, while the output consists of results from anova decomposition
and indices's corresponding to each result (refer `Index
Mapping <#index mapping>`__).

For a given experiment the anchored point is per-selected and data is
generated on the hyper-lines and hyper-planes with respect to the
anchored point. For this reason, supplying an anchored point along with
the data would not make much sense. As data may not be available for
corresponding to this random selection of this anchor point. Hence, the
mean, 1\ :sup:`st` order and 2\ :sup:`nd` order terms are directly
supplied. The strategy is as follows with reference to `Figure
3 <#anchor-25>`__ in 4 dimensions:

-  For the mean or anchored point term: Supply the data as the 1
   dimensional array.
-  1\ :sup:`st` order experimental data: In form of three dimensional
   array. Along dimension 1 of this array supply the data obtained form
   a single simulation or experiment for the corresponding term and
   quadrature point. Dimension 2 of this array: increment the quadrature
   points for the corresponding term. Dimension 3 of this array: number
   of first order terms. As an example: at location [10, 1, 2] has the
   11\ :sup:`th` CFD grid point data is stored for f3(c, q\ :sub:`1,` c,
   c). Notice that, in python indices's begin with zero. Shape (*net
   grid pts x net quad pts x net 1*\ :sup:`st`\ *order terms*)
-  2\ :sup:`nd` order experimental data: Similar strategy as per
   1\ :sup:`st` order terms, with only difference in 3\ :sup:`rd`
   dimension instead of 1\ :sup:`st` order terms, we have 2\ :sup:`nd`
   order terms in a chronological fashion.

.. figure:: Pictures/100000000000021B000001D53D1D680927E37EB7.png
   :alt: 
   Figure 3: Array for 1st and 2nd order terms (Input)
   :width: 3.2984in
   :height: 2.8701in

   Figure 3: Array for 1st and 2nd order terms (Input)

The output c-ANOVA decomposition is a two dimensional array. With
reference to `Figure 4 <#anchor-26>`__, each column is an individual
solution where along the 1\ :sup:`st` dimension of this output array we
have grid points data. By calling a particular row number the
corresponding solution can be called, refer `Figure 2 <#anchor-24>`__.
It is to be noted that this output strategy is being used for all the
input and outputs in subsequent sections, unless specifically mentioned.

.. figure:: Pictures/10000000000001EE0000013F1885FFC8228030DB.png
   :alt: 
   Figure 4: c-ANOVA decomposition output array
   :width: 3.5055in
   :height: 2.352in

   Figure 4: c-ANOVA decomposition output array

3.3 POD and Krigging

Proper Orthogonal Decomposition is performed by the script, “pod.py”,
while krigging is done using “krigging.py”. As mentioned earlier the
input and output for both of these techniques follow the discussion of
ANOVA output (refer `Figure 4 <#anchor-26>`__). Additional flexibility
is provided; in POD, the technique is robust enough to find the
orthogonal basis (refer `(7) <#anchor-8>`__) for any random data,
however it is recommended for the data to be in the form described in
`Figure 4 <#anchor-26>`__. It is to be noted that the output is as per
the input. For example, if we have a different indexing strategy, the
output would now correspond to this new indexing system. Nonetheless, it
still needs to be an array in 2 dimensions only.

While, for krigging; the flexibility provided algorithmically than
arising mathematically. Here, as long as the indice’s array is supplied
along with the data, the python script will identify and perform
krigging accordingly. However, the indexing system along with the data
needs to be in the form described in the earlier sections. We now
introduce, how krigging is being performed from 1 dimension to 2
dimensions, and this discussion can be extend to *N* – dimensions.

Consider an example; there are two quadrature points in two dimensions
each, let (0.25, 0.75) and (-0.25, -0.75) be those quadrature points in
dimension 1 and 2 respectively. Now, we require the krigging
interpolated data at the midpoint of these quadrature points in each
dimension. Hence, our *discretization samples* = 3. In general,
(*discretization samples = number of required point data + 2)* for each
dimension. Referring `Figure 5 <#anchor-27>`__, we start krigging in the
first dimension. The interpolation is carried out for the point (0.5,
-0.25), here we need the data at (0.25, -0.25) and (0.75, -0.25); then
for (0.5, -0.75), here the required data is at (0.25, -0.75) and (0.75,
-0.75) and so on.

In general, we fix all the quadrature points in all dimensions except
the dimension where the interpolation is happening. For example,
consider an *N*-dimensional case, where the interpolation is happening
at dimension 1 then, (discretization, c1,c2, .. ,cN) would require data
from (all quad points, c1, c2, ..,cN), here (c1, … cN) are points fixed
in space.

.. figure:: Pictures/10000000000002F50000014432009FC268F7F830.png
   :alt: 
   Figure 5: Krigging: Interpolating data in each dimension at a time
   :width: 4.8717in
   :height: 2.0846in

   Figure 5: Krigging: Interpolating data in each dimension at a time

It is to be noted that `Kernel <#anchor-28>`__ is only a feature of
Krigging. Also, by default it takes uniform samples, unless samples are
provided (refer `Sampling <#anchor-29>`__)

3.4 Software Architecture

In the previous sections; the main files input – output functionality
was explained. In this section, the software architecture is layed-out.
It is to be noted, due to the objected oriented nature of all these
scripts; any script or method can be called, in any order. From `Figure
6 <#anchor-30>`__, the “cAPK – main” python script is the control file.
The experimental data for performing an UQ study is built in the
“case.py” script as per the above discussions. This script is then
called by the main script and supplied to specific UQ – technique as
required in the study. Data from one methodology can be supplied to any
other methodology as long as the indices's are provided as well (refer
`Index Mapping <#index mapping>`__).

It is to be noted that for Krigging, by default it takes uniform
sampling, generated by “samples.py”, unless the samples are provided
from any other sampling strategy. In order to generate “sobol sequence”,
the “samples.py” is needed to be called. The samples now generated using
this sampling strategy can be fed into the anyother UQ methodology,
where-ever applicable.

|
Figure 6: Software Architecture|\ 3.5 Miscellaneous

3.5.1 Sampling and Sobol sequence generator

The “sample.py” generates samples as per uniform or sobol sequence. In
the uniform strategy, the end points are also returned. The “sobol.py”
is used to generate samples using sobol sequence. It is recommended to
add additional sampling strategy in this file.

3.5.2 Kernel

In this UQ study kernel is only a feature of krigging. However, due to
object orientated nature of the script, it can be called any other
script. In he “kernel.py” python script, currently only two kernels are
implemented. Namely, the “square expectational” and “polynomial cubic
splines” as per equation `(13) <#anchor-17>`__ and `(14) <#anchor-18>`__
respectively. Again, it is recommended to add additional kernels in this
script. Moreover, python’s “sckit-learn” library has additional kernels
to use.

4. Numerical Investigations

For our numerical investigation a case of a Lid Driven cavity with 4
uncertain dimensions as outlined in `Figure 7 <#anchor-31>`__. For
carrying out this test we have u1 = [0.25, 0.75], u2 = [-0.25, -0.75],
v1 = [0.25, 0.75] and v2 = [-0.25, -0.75].

.. figure:: Pictures/10000000000001CC00000185EA4D922455E526A4.png
   :alt: 
   Figure 7: Lid Driven cavity with 4 uncertain dimensions.
   :width: 2.5055in
   :height: 2.1189in

   Figure 7: Lid Driven cavity with 4 uncertain dimensions.

4.1 ANOVA Decomposition

The anchored ANOVA decomposition is carried out in 4 dimensions with the
truncation dimension (*v*) = 2. From `Figure A.1 <#anchor-32>`__,
qualitatively speaking the results are well converged for both ANOVA
decomposition with truncation dimension *v* = 1, 2; with *v = 2*
performing better compared to *v = 1*. From literature it is well known
fact the *v =*\ 1 captures the maximum variance in the data, hence, our
convergence of *v = 1* result is well justified. However, the
interesting part was to look at the quantitative data. In order to
determine that the error from ANOVA decomposition when compared to CFD
results, the absolute difference between the two was computed as per
`(15) <#anchor-33>`__.

== ========= ====
\  |image15| (15)
== ========= ====

It was found that certain low velocity regions did not converge well,
while the regions having higher values did. Initial investigation was
about Python’s capability for handling low or near zero value; as the
default data type in Python is double and after algebraic operations,
near zero values are represented as (+/-) 1e-8. In this effect a Zero
handler method (not shown here) was created; where detection of such
values it would force it take reasonably low value. However, the
relative error did not change much, as both the ANOVA and CFD result
would scale itself to produce the same difference. On the contrary,
(generality speaking) this behavior of Python provides a fail safe
algorithm, where division by zero would not crash the computation and
having 1e-8 value would supply a very negligible error to the over all
solution. Hence, the we eliminate the algorithmic possibility of Python
contributing towards the non-convergent regions. This leaves us with two
other possibilities:

-  Bad choice of anchor point
-  High variance in the data – implying the need for higher order terms.

Either way, if high variance is the reason, it should be well reflected
in our present data. First, the mean deviation of CFD results was
determined as per `(16) <#anchor-35>`__.

== ========= ====
\  |image16| (16)
== ========= ====

The assumption of high variance was true in the regions of highest
error. Contrastingly, the variance around the boundary was higher too,
which is a well converged region. This convergence along the boundary
region can be attributed to well converged c-ANOVA results with
1\ :sup:`st` order terms as seen in `Figure A.1 <#anchor-32>`__.
Implying an higher variance in second order terms. Hence, an estimator
for computation has been formulated inspired by Sobol’s work given in
`(17) <#anchor-37>`__

== ========= ====
\  |image17| (17)
== ========= ====

The results of `(17) <#anchor-37>`__ shows the highest variance in the
regions of highest error (`Figure A.1 <#anchor-32>`__). The assumption
of variance based error can be now be concluded in the second order
terms. Finally, to determine the relative error in ANOVA decomposition
the use of `(15) <#anchor-33>`__ is not feasible in all cases, owing to
the fact of unavailability of high dimensional CFD simulation data. An
additional estimator was formulated based on c-ANOVA decomposition only,
given in `(18) <#anchor-39>`__, where *f’* is the ANOVA decomposition
results with truncation dimension = 1, while *f’*\ ’ is the results with
truncation dimension = 2.

== ========= ====
\  |image18| (18)
== ========= ====

From `Figure A.1 <#anchor-32>`__, there is a close resemblance of this
new estimator with `(15) <#anchor-33>`__, owing to the fact the ANOVA
decomposition reasonably converged with truncation dimension = 1.
Furthermore, this new estimator is not limited to this case only, as
literature shows us that maximum variance in the data is captured by
1\ :sup:`st` order terms, which is our case too. In conclusion,
interpretation from variance order 2 `(17) <#anchor-37>`__ and ANOVA
Error estimator `(18) <#anchor-39>`__ can help us identify regions of
high error without the need for additional CFD computations.

4.2 Krigging

At first the effect of two different kernel (`(13) <#anchor-17>`__ and
`(14) <#anchor-18>`__) is investigated for two different equations. From
`Figure 8 <#anchor-41>`__, the square exponential kernel is not able to
capture the quadratic equation (*y = x^2*)\ *,*\ while the “Polynomial
Cubic Spline” kernel does. Furthermore, we look at the function (*y =
x*cos(x)*) with extrapolation. As expected, in the extrapolated region
the variance or the error is too high. At the interpolated regions the
square exponential kernel behaves poorly too with a high variance.

========= ========================= ===============================
\         Square exponential Kernel Polynomial Cubic Spline Kernel 
|image19| |image20|                 |image21|
|image22| |image23|                 |image24|
========= ========================= ===============================

Figure 8: Results from two different Kernels (`(13) <#anchor-17>`__ and
`(14) <#anchor-18>`__)

Furthermore, investigation of krigging applied to CFD simulations are
carried out in determining the relative error of both of these kernel.
As expected the error in any given individual solution is uniform and
found to be in the range of [0%, 40%]. This high error can be attributed
to poor performance of the “square exponential” kernel. Selected results
of krigging applied to CFD are given in `Figure 9 <#anchor-42>`__, where
the discretization sample = 3 (refer `Figure 5 <#anchor-27>`__ for index
map). Qualitatively speaking, the results from Krigging using
“polynomial cubic spline” kernel are very close to actual case, while
the “square exponential” kernel are overestimated.

========= =========
|image25| |image26|
|image27| |image28|
|image29| |image30|
|image31| |image32|
|image33| |image34|
|image35| |image36|
========= =========

Figure 9: Selected results from Krigging applied to CFD results with
discretization sample = 3 (total number of results = 3^4 = 81), obtained
using two kernels (refer `(13) <#anchor-17>`__ and
`(14) <#anchor-18>`__)

For the reasons outlined above (with reference to `Figure
8 <#anchor-41>`__ and `Figure 9 <#anchor-42>`__). The “polynomial cubic
spline” kernel would was used for our further study. In `Figure
B.1 <#anchor-43>`__, krigging was applied to CFD and ANOVA (*v =
2*)\ *,*\ qualitatively shows great similarity. An interesting pattern
was observed when computing their absolute difference. This error arises
from ANOVA decomposition, where a similar error paterns are observed
(refer `Figure A.1 <#anchor-32>`__).

4.3 Proper Orthogonal Decomposition

The results of POD were highly erroneous. The failure of POD is due to
un-converged correlation matrix. In this effect upon increasing the
samples for POD from Krigging was investigated. However, there has been
an error reduction upon increase in samples but very slowly and required
very high number of samples, as seen `Figure 10 <#anchor-44>`__. Many of
the POD results had much higher error than aforementioned figure (not
shown here).

========= =========
|image37| |image38|
========= =========

Figure 10: Relative error of POD, normalized using krigging data,

(from left to right) total number of samples = 625 and 6561.

Upon a closer look at the results, it was observed, POD averaged out the
solution and identified the regions of high velocity (`Figure
11 <#anchor-45>`__). This observation is consistent with many
literatures where POD is primarily used to identify structures or
regions of high kinetic energy in a given flow. The last result in
`Figure 11 <#anchor-45>`__ is the at the same sampling point as `Figure
10 <#anchor-44>`__, which is [0.75, -0.75, 0.75, -0.75].

========= =========
|image39| |image40|
|image41| |image42|
|image43| |image44|
|image45| |image46|
|image47| |image48|
|image49| |image50|
========= =========

Figure 11: CFD results and their POD on right

4.4 Sobol indices's

Initial samples per dimension = 3, results shown for samples per
dimension = 4

Intital Error = 100.38595089174746

Intital Error = 56.02111144209835

Intital Error = 70.87682286827632

Intital Error = 57.80812238439015

== =========================== =========================
\  Sobol indices's using ANOVA Sobol indices's using CFD
S1 |image51|                   |image52|
S2 |image53|                   |image54|
S3 |image55|                   |image56|
S4 |image57|                   |image58|
== =========================== =========================

Intital Error = 440.41568265715426

Intital Error = 32.81866216922758

Intital Error = 5.660749534046695

Intital Error = 34.20805733963393

solution not converged with samples per dimesnion = 6 & total samples in
all dimensions = 1296

incrementing samples in each dimension

Error Reduction to = 119.58576424301738 with samples per dim = 7

Error Reduction to = 1.195039822065792 with samples per dim = 7

Error Reduction to = 38.02697538071071 with samples per dim = 7

Error Reduction to = 1.4243297314201684 with samples per dim = 7

solution not converged with samples per dimesnion = 7 & total samples in
all dimensions = 2401

incrementing samples in each dimension

Error Reduction to = 11.411923813409057 with samples per dim = 8

Error Reduction to = 10.869029032949072 with samples per dim = 8

Error Reduction to = 3.4281133096043015 with samples per dim = 8

Error Reduction to = 11.709699782885695 with samples per dim = 8

solution not converged with samples per dimesnion = 8 & total samples in
all dimensions = 4096

incrementing samples in each dimension

Error Reduction to = 14.461142013119678 with samples per dim = 9

Error Reduction to = 0.3059322793530657 with samples per dim = 9

Error Reduction to = 5.959893208171849 with samples per dim = 9

Error Reduction to = 0.16132129815380492 with samples per dim = 9

solution not converged with samples per dimesnion = 9 & total samples in
all dimensions = 6561

incrementing samples in each dimension

Error Reduction to = 57.952285840422036 with samples per dim = 10

Error Reduction to = 27.07896684426755 with samples per dim = 10

Error Reduction to = 9.17105380620675 with samples per dim = 10

Error Reduction to = 28.998532566926887 with samples per dim = 10

solution not converged with samples per dimesnion = 10 & total samples
in all dimensions = 10000

incrementing samples in each dimension

Error Reduction to = 23.55757030618551 with samples per dim = 11

Error Reduction to = 2.1425882533838876 with samples per dim = 11

Error Reduction to = 7.87546745461422 with samples per dim = 11

Error Reduction to = 1.9916640583241414 with samples per dim = 11

solution not converged with samples per dimesnion = 11 & total samples
in all dimensions = 14641

incrementing samples in each dimension

Error Reduction to = 14.509662855918455 with samples per dim = 12

Error Reduction to = 0.8446846919961216 with samples per dim = 12

Error Reduction to = 3.1949545258207652 with samples per dim = 12

Error Reduction to = 0.6226657623697935 with samples per dim = 12

solution not converged with samples per dimesnion = 12 & total samples
in all dimensions = 20736

incrementing samples in each dimension

Error Reduction to = 24.084316044180675 with samples per dim = 13

Error Reduction to = 11.527237473460648 with samples per dim = 13

Error Reduction to = 14.951215862751718 with samples per dim = 13

Error Reduction to = 12.4109584804769 with samples per dim = 13

solution not converged with samples per dimesnion = 13 & total samples
in all dimensions = 28561

incrementing samples in each dimension

Error Reduction to = 18.14990909739811 with samples per dim = 14

Error Reduction to = 0.3270795406638389 with samples per dim = 14

Error Reduction to = 4.102562251887288 with samples per dim = 14

Error Reduction to = 0.5417576381935958 with samples per dim = 14

solution not converged with samples per dimesnion = 14 & total samples
in all dimensions = 38416

incrementing samples in each dimension

Error Reduction to = 47.34582945035988 with samples per dim = 15

Error Reduction to = 7.8581807557544785 with samples per dim = 15

Error Reduction to = 20.51812556891884 with samples per dim = 15

Error Reduction to = 8.887133054877111 with samples per dim = 15

solution not converged with samples per dimesnion = 15 & total samples
in all dimensions = 50625

incrementing samples in each dimension

Error Reduction to = 17.099131371475664 with samples per dim = 16

Error Reduction to = 9.403106374672527 with samples per dim = 16

Error Reduction to = 0.12386737353250678 with samples per dim = 16

Error Reduction to = 10.019744591774566 with samples per dim = 16

Below results for Sobol indices's (S1, S2, S3 and S4 in a clockwise
sense), mean and variance for samples per dimension = 17. Total samples
= 83521.

Expected Error

Error Reduction to = 17.099131371475664 with samples per dim = 16

Error Reduction to = 9.403106374672527 with samples per dim = 16

Error Reduction to = 0.12386737353250678 with samples per dim = 16

Error Reduction to = 10.019744591774566 with samples per dim = 16

========= =========
|image59| |image60|
|image61| |image62|
========= =========

========= =========
Mean      Variance
|image63| |image64|
========= =========

Bibliography

Appendix A: ANOVA Results

Figure A.1: Results from CFD, Error of ANOVA decomposition, *v = 2*
(refer `(15) <#anchor-33>`__) , ANOVA Error Estimator
`(18) <#anchor-39>`__, Variance of 2\ :sup:`nd` Order terms (refer
`(17) <#anchor-37>`__), ANOVA decomposition (*v =1*) and ANOVA
decomposition (*v =2*), in a clockwise sense. Indices's as per `Figure
2 <#anchor-24>`__

========= ========= =========
|image65| |image66| |image67|
|image68| |image69| |image70|
========= ========= =========

========= ========= =========
|image71| |image72| |image73|
|image74| |image75| |image76|
========= ========= =========

========= ========= =========
|image77| |image78| |image79|
|image80| |image81| |image82|
========= ========= =========

========= ========= =========
|image83| |image84| |image85|
|image86| |image87| |image88|
========= ========= =========

========= ========= =========
|image89| |image90| |image91|
|image92| |image93| |image94|
========= ========= =========

========= ========= ==========
|image95| |image96| |image97|
|image98| |image99| |image100|
========= ========= ==========

========== ========== ==========
|image101| |image102| |image103|
|image104| |image105| |image106|
========== ========== ==========

========== ========== ==========
|image107| |image108| |image109|
|image110| |image111| |image112|
========== ========== ==========

========== ========== ==========
|image113| |image114| |image115|
|image116| |image117| |image118|
========== ========== ==========

========== ========== ==========
|image119| |image120| |image121|
|image122| |image123| |image124|
========== ========== ==========

========== ========== ==========
|image125| |image126| |image127|
|image128| |image129| |image130|
========== ========== ==========

========== ========== ==========
|image131| |image132| |image133|
|image134| |image135| |image136|
========== ========== ==========

========== ========== ==========
|image137| |image138| |image139|
|image140| |image141| |image142|
========== ========== ==========

========== ========== ==========
|image143| |image144| |image145|
|image146| |image147| |image148|
========== ========== ==========

========== ========== ==========
|image149| |image150| |image151|
|image152| |image153| |image154|
========== ========== ==========

========== ========== ==========
|image155| |image156| |image157|
|image158| |image159| |image160|
========== ========== ==========

Appendix B: Krigging Results

Figure B.1: From left to right, Krigging using “Polynomial Cubic Spline”
kernel with discretization sample = 3; applied to CFD & ANOVA; and their
absolute difference (*only the 1*\ :sup:`st`\ *20 results are shown here
out of 81*).

========== ========== ==========
|image161| |image162| |image163|
|image164| |image165| |image166|
========== ========== ==========

========== ========== ==========
|image167| |image168| |image169|
|image170| |image171| |image172|
========== ========== ==========

========== ========== ==========
|image173| |image174| |image175|
|image176| |image177| |image178|
========== ========== ==========

========== ========== ==========
|image179| |image180| |image181|
|image182| |image183| |image184|
========== ========== ==========

========== ========== ==========
|image185| |image186| |image187|
|image188| |image189| |image190|
========== ========== ==========

========== ========== ==========
|image191| |image192| |image193|
|image194| |image195| |image196|
========== ========== ==========

========== ========== ==========
|image197| |image198| |image199|
|image200| |image201| |image202|
========== ========== ==========

========== ========== ==========
|image203| |image204| |image205|
|image206| |image207| |image208|
========== ========== ==========

========== ========== ==========
|image209| |image210| |image211|
|image212| |image213| |image214|
========== ========== ==========

========== ========== ==========
|image215| |image216| |image217|
|image218| |image219| |image220|
========== ========== ==========

.. |image0| image:: ./ObjectReplacements/Object 1
   :width: 5.5126in
   :height: 0.3201in
.. |image1| image:: ./ObjectReplacements/Object 2
   :width: 3.5283in
   :height: 0.9638in
.. |image2| image:: ./ObjectReplacements/Object 3
   :width: 4.9563in
   :height: 0.5744in
.. |image3| image:: ./ObjectReplacements/Object 4
   :width: 4.1346in
   :height: 0.3201in
.. |image4| image:: ./ObjectReplacements/Object 5
   :width: 3.3602in
   :height: 0.2689in
.. |image5| image:: ./ObjectReplacements/Object 6
   :width: 1.0839in
   :height: 0.2516in
.. |image6| image:: ./ObjectReplacements/Object 7
   :width: 2.4925in
   :height: 0.6047in
.. |image7| image:: ./ObjectReplacements/Object 8
   :width: 1.5835in
   :height: 0.4126in
.. |image8| image:: ./ObjectReplacements/Object 9
   :width: 1.8575in
   :height: 0.4126in
.. |image9| image:: ./ObjectReplacements/Object 10
   :width: 1.0701in
   :height: 0.3937in
.. |image10| image:: ./ObjectReplacements/Object 11
   :width: 1.1953in
   :height: 0.2008in
.. |image11| image:: ./ObjectReplacements/Object 12
   :width: 1.3547in
   :height: 0.2689in
.. |image12| image:: ./ObjectReplacements/Object 13
   :width: 2.6917in
   :height: 0.3937in
.. |image13| image:: ./ObjectReplacements/Object 14
   :width: 4.9827in
   :height: 1.2772in
.. |
Figure 6: Software Architecture| image:: Pictures/1000000000000304000002EEA809ABDB9E5249DC.png
   :width: 5.1862in
   :height: 5.0382in
.. |image15| image:: ./ObjectReplacements/Object 15
   :width: 3.639in
   :height: 0.2008in
.. |image16| image:: ./ObjectReplacements/Object 16
   :width: 2.872in
   :height: 0.2638in
.. |image17| image:: ./ObjectReplacements/Object 17
   :width: 2.0492in
   :height: 0.4126in
.. |image18| image:: ./ObjectReplacements/Object 18
   :width: 2.9398in
   :height: 0.2008in
.. |image19| image:: Pictures/100000000000002400000087F8E1929DD2004606.png
   :width: 0.3752in
   :height: 1.4063in
.. |image20| image:: Pictures/100002010000017A00000108ED01FDB3955129F0.png
   :width: 3.0236in
   :height: 2.111in
.. |image21| image:: Pictures/100002010000017A000001082423246678279C51.png
   :width: 3.0217in
   :height: 2.1098in
.. |image22| image:: Pictures/1000000000000021000000A47333C0A9C0BE66E8.png
   :width: 0.3437in
   :height: 1.7083in
.. |image23| image:: Pictures/1000020100000176000001081D6B85160D02A92B.png
   :width: 3.0236in
   :height: 2.1339in
.. |image24| image:: Pictures/100002010000017600000108D23B838C09665499.png
   :width: 3.0217in
   :height: 2.1327in
.. |image25| image:: Pictures/1000020100000176000001080CACFA0B8B31D1E1.png
   :width: 3.2701in
   :height: 2.3083in
.. |image26| image:: Pictures/100002010000017C0000010811C58E84B33FE5A5.png
   :width: 3.2701in
   :height: 2.2717in
.. |image27| image:: Pictures/100002010000017600000108F3A1825656E40C39.png
   :width: 3.2701in
   :height: 2.3083in
.. |image28| image:: Pictures/100002010000017C00000108BC493B4D7684299D.png
   :width: 3.2701in
   :height: 2.2717in
.. |image29| image:: Pictures/10000201000001760000010801CBF106CF60B23B.png
   :width: 3.2701in
   :height: 2.3083in
.. |image30| image:: Pictures/100002010000017C0000010833C12EE466A44543.png
   :width: 3.2701in
   :height: 2.2717in
.. |image31| image:: Pictures/100002010000017600000108376DAD2ED4FB4EFC.png
   :width: 3.2701in
   :height: 2.3083in
.. |image32| image:: Pictures/100002010000017C0000010879627956235C2A9A.png
   :width: 3.2701in
   :height: 2.2717in
.. |image33| image:: Pictures/1000020100000176000001088CCABD48D78D0540.png
   :width: 3.2701in
   :height: 2.3083in
.. |image34| image:: Pictures/100002010000017C000001088DD766557BD1FC4C.png
   :width: 3.2701in
   :height: 2.2717in
.. |image35| image:: Pictures/1000020100000176000001086DC67437D527F310.png
   :width: 3.2701in
   :height: 2.3083in
.. |image36| image:: Pictures/100002010000017C000001084D65285875965179.png
   :width: 3.2701in
   :height: 2.2717in
.. |image37| image:: Pictures/100002010000017000000108B02D5A1300F46299.png
   :width: 2.1543in
   :height: 1.5453in
.. |image38| image:: Pictures/100002010000017000000108B02D5A1300F46299.png
   :width: 2.1543in
   :height: 1.5453in
.. |image39| image:: Pictures/100002010000017600000108BDC57E4E7AD43124.png
   :width: 3.2957in
   :height: 2.3264in
.. |image40| image:: Pictures/100002010000017800000108FE222EFF727BAA49.png
   :width: 3.2701in
   :height: 2.2957in
.. |image41| image:: Pictures/1000020100000176000001085A9AB95C72B998AC.png
   :width: 3.2957in
   :height: 2.3264in
.. |image42| image:: Pictures/100002010000017800000108CE15A4A93ED5E523.png
   :width: 3.2701in
   :height: 2.2957in
.. |image43| image:: Pictures/100002010000017600000108BAC70A42E17C52FF.png
   :width: 3.2957in
   :height: 2.3264in
.. |image44| image:: Pictures/100002010000017800000108CE844069DFE90BDC.png
   :width: 3.2701in
   :height: 2.2957in
.. |image45| image:: Pictures/1000020100000176000001085416125C886CD6F6.png
   :width: 3.2957in
   :height: 2.3264in
.. |image46| image:: Pictures/100002010000017800000108A60B75B436BDC5F9.png
   :width: 3.2701in
   :height: 2.2957in
.. |image47| image:: Pictures/10000201000001760000010830C46E45BCE1D8DD.png
   :width: 3.2957in
   :height: 2.3264in
.. |image48| image:: Pictures/100002010000017800000108DFBCB654FD5FC2DE.png
   :width: 3.2701in
   :height: 2.2957in
.. |image49| image:: Pictures/100002010000017600000108E2B2E86E531C8C4B.png
   :width: 3.2957in
   :height: 2.3264in
.. |image50| image:: Pictures/100002010000017800000108C413800F82F0FB0E.png
   :width: 3.2701in
   :height: 2.2957in
.. |image51| image:: Pictures/1000020100000170000000FCEFFB712BF6972002.png
   :width: 3.1138in
   :height: 2.1319in
.. |image52| image:: Pictures/1000020100000170000000FCEAEC1E29837AB1F3.png
   :width: 3.111in
   :height: 2.1299in
.. |image53| image:: Pictures/1000020100000170000000FC2EDDA069E6E9BBD2.png
   :width: 3.1138in
   :height: 2.1319in
.. |image54| image:: Pictures/1000020100000170000000FC56DEAC80B5D735E7.png
   :width: 3.111in
   :height: 2.1299in
.. |image55| image:: Pictures/1000020100000170000000FCFB939D7AA8365256.png
   :width: 3.1138in
   :height: 2.1319in
.. |image56| image:: Pictures/1000020100000170000000FC0AA36F774C19C800.png
   :width: 3.111in
   :height: 2.1299in
.. |image57| image:: Pictures/1000020100000170000000FC3834D1A3D9DB9E8A.png
   :width: 3.1138in
   :height: 2.1319in
.. |image58| image:: Pictures/1000020100000170000000FCBB4530F52037089A.png
   :width: 3.111in
   :height: 2.1299in
.. |image59| image:: Pictures/1000020100000170000000FCB06B0C9A53E78A26.png
   :width: 3.2701in
   :height: 2.239in
.. |image60| image:: Pictures/1000020100000170000000FC519A7A35EF7A8CF0.png
   :width: 3.2701in
   :height: 2.239in
.. |image61| image:: Pictures/1000020100000170000000FC5127AADA90DC05A7.png
   :width: 3.2701in
   :height: 2.239in
.. |image62| image:: Pictures/1000020100000170000000FC3465423183C62114.png
   :width: 3.2701in
   :height: 2.239in
.. |image63| image:: Pictures/1000020100000176000000FC10362B00983EDD87.png
   :width: 3.2701in
   :height: 2.2035in
.. |image64| image:: Pictures/1000020100000183000000FC64F1CC6BA98E9430.png
   :width: 3.2701in
   :height: 2.1291in
.. |image65| image:: Pictures/100002010000017600000108DAFE562A432F46D1.png
   :width: 3.2957in
   :height: 2.3264in
.. |image66| image:: Pictures/100002010000017C000001084F4FCAF9A291A614.png
   :width: 3.2965in
   :height: 2.2902in
.. |image67| image:: Pictures/10000201000001760000010882C9383402DD67D7.png
   :width: 3.2965in
   :height: 2.3272in
.. |image68| image:: Pictures/1000020100000180000001084E84F646DF5E727E.png
   :width: 3.2957in
   :height: 2.2661in
.. |image69| image:: Pictures/100002010000018000000108DC367F0F697B2D4A.png
   :width: 3.2965in
   :height: 2.2661in
.. |image70| image:: Pictures/100002010000017600000108A22E3CC6B6990F78.png
   :width: 3.2965in
   :height: 2.3272in
.. |image71| image:: Pictures/1000020100000176000001081BC81EBE57948AE8.png
   :width: 3.2957in
   :height: 2.3264in
.. |image72| image:: Pictures/100002010000017C0000010819BC40D4530E504C.png
   :width: 3.2965in
   :height: 2.2902in
.. |image73| image:: Pictures/10000201000001760000010818AE0238766960C7.png
   :width: 3.2965in
   :height: 2.3272in
.. |image74| image:: Pictures/100002010000018000000108348563C5D2981F13.png
   :width: 3.2957in
   :height: 2.2661in
.. |image75| image:: Pictures/100002010000018000000108AA3906AAD7D5EC4C.png
   :width: 3.2965in
   :height: 2.2661in
.. |image76| image:: Pictures/1000020100000176000001085A97B662219C6829.png
   :width: 3.2965in
   :height: 2.3272in
.. |image77| image:: Pictures/100002010000017600000108EB0707C5621D544E.png
   :width: 3.2957in
   :height: 2.3264in
.. |image78| image:: Pictures/100002010000017C00000108DE7D1110FA306591.png
   :width: 3.2965in
   :height: 2.2902in
.. |image79| image:: Pictures/1000020100000176000001082621294197C8B04C.png
   :width: 3.2965in
   :height: 2.3272in
.. |image80| image:: Pictures/100002010000018000000108A45B94D4D1592939.png
   :width: 3.2957in
   :height: 2.2661in
.. |image81| image:: Pictures/1000020100000180000001089E526F2FE1272167.png
   :width: 3.2965in
   :height: 2.2661in
.. |image82| image:: Pictures/100002010000017600000108D6DBF403400A9362.png
   :width: 3.2965in
   :height: 2.3272in
.. |image83| image:: Pictures/1000020100000176000001085A17AE60391C8F79.png
   :width: 3.2957in
   :height: 2.3264in
.. |image84| image:: Pictures/100002010000017C00000108A791CC151877DB6C.png
   :width: 3.2965in
   :height: 2.2902in
.. |image85| image:: Pictures/100002010000017600000108B55E458EC9139145.png
   :width: 3.2965in
   :height: 2.3272in
.. |image86| image:: Pictures/100002010000018000000108123C9B2CABE1F42F.png
   :width: 3.2957in
   :height: 2.2661in
.. |image87| image:: Pictures/1000020100000180000001084478F24727EEEFA1.png
   :width: 3.2965in
   :height: 2.2661in
.. |image88| image:: Pictures/100002010000017600000108A1F4A138695D7D8B.png
   :width: 3.2965in
   :height: 2.3272in
.. |image89| image:: Pictures/100002010000017600000108FD9BCA0F05452461.png
   :width: 3.2957in
   :height: 2.3264in
.. |image90| image:: Pictures/100002010000017C00000108B6207CD26363EC01.png
   :width: 3.2965in
   :height: 2.2902in
.. |image91| image:: Pictures/100002010000017600000108C9544209CD371966.png
   :width: 3.2965in
   :height: 2.3272in
.. |image92| image:: Pictures/100002010000018000000108B9FBD1907FA5C306.png
   :width: 3.2957in
   :height: 2.2661in
.. |image93| image:: Pictures/10000201000001800000010802F261BCA38CB68A.png
   :width: 3.2965in
   :height: 2.2661in
.. |image94| image:: Pictures/10000201000001760000010880713FBE58EC5959.png
   :width: 3.2965in
   :height: 2.3272in
.. |image95| image:: Pictures/10000201000001760000010823DEB8DD817722B9.png
   :width: 3.2957in
   :height: 2.3264in
.. |image96| image:: Pictures/100002010000017C0000010892746B15BB083F25.png
   :width: 3.2965in
   :height: 2.2902in
.. |image97| image:: Pictures/10000201000001760000010816AEC284D125F491.png
   :width: 3.2965in
   :height: 2.3272in
.. |image98| image:: Pictures/1000020100000180000001088DB11CF84563F65C.png
   :width: 3.2957in
   :height: 2.2661in
.. |image99| image:: Pictures/10000201000001800000010899792F009B305A7C.png
   :width: 3.2965in
   :height: 2.2661in
.. |image100| image:: Pictures/100002010000017600000108A04DE46C229EF76C.png
   :width: 3.2965in
   :height: 2.3272in
.. |image101| image:: Pictures/100002010000017600000108332B1FF41A262738.png
   :width: 3.2957in
   :height: 2.3264in
.. |image102| image:: Pictures/100002010000017C00000108CC879CA18BED9EDF.png
   :width: 3.2965in
   :height: 2.2902in
.. |image103| image:: Pictures/100002010000017600000108FA9454F837151E56.png
   :width: 3.2965in
   :height: 2.3272in
.. |image104| image:: Pictures/10000201000001800000010881ACCD06A330C8A9.png
   :width: 3.2957in
   :height: 2.2661in
.. |image105| image:: Pictures/100002010000018000000108E424BC05E63384EC.png
   :width: 3.2965in
   :height: 2.2661in
.. |image106| image:: Pictures/10000201000001760000010867F3954308553DE5.png
   :width: 3.2965in
   :height: 2.3272in
.. |image107| image:: Pictures/100002010000017600000108A758CB8F882121C2.png
   :width: 3.2957in
   :height: 2.3264in
.. |image108| image:: Pictures/100002010000017C00000108E89BFFEDE8322079.png
   :width: 3.2965in
   :height: 2.2902in
.. |image109| image:: Pictures/100002010000017600000108B42BA59A2FA0EDA7.png
   :width: 3.2965in
   :height: 2.3272in
.. |image110| image:: Pictures/10000201000001800000010818E30A4F4EEB778E.png
   :width: 3.2957in
   :height: 2.2661in
.. |image111| image:: Pictures/1000020100000180000001085F81A33671576026.png
   :width: 3.2965in
   :height: 2.2661in
.. |image112| image:: Pictures/10000201000001760000010870D3D8EA4190FA16.png
   :width: 3.2965in
   :height: 2.3272in
.. |image113| image:: Pictures/100002010000017600000108157D15F07A8C23E5.png
   :width: 3.2957in
   :height: 2.3264in
.. |image114| image:: Pictures/100002010000017C000001081914E792E46013E6.png
   :width: 3.2965in
   :height: 2.2902in
.. |image115| image:: Pictures/1000020100000176000001086AB04F9CEBF5699E.png
   :width: 3.2965in
   :height: 2.3272in
.. |image116| image:: Pictures/1000020100000180000001083DB1BB859A994948.png
   :width: 3.2957in
   :height: 2.2661in
.. |image117| image:: Pictures/1000020100000180000001080E9EA922F099EE6B.png
   :width: 3.2965in
   :height: 2.2661in
.. |image118| image:: Pictures/100002010000017600000108FF0D3C3C60FCC780.png
   :width: 3.2965in
   :height: 2.3272in
.. |image119| image:: Pictures/100002010000017600000108C971C4CDE2943C4B.png
   :width: 3.2957in
   :height: 2.3264in
.. |image120| image:: Pictures/100002010000017C0000010879EE37FA3019E7F2.png
   :width: 3.2965in
   :height: 2.2902in
.. |image121| image:: Pictures/100002010000017600000108AB5CD416DBFFA132.png
   :width: 3.2965in
   :height: 2.3272in
.. |image122| image:: Pictures/100002010000018000000108A7DAF59ECD49E747.png
   :width: 3.2957in
   :height: 2.2661in
.. |image123| image:: Pictures/100002010000018000000108F24CF982527F9505.png
   :width: 3.2965in
   :height: 2.2661in
.. |image124| image:: Pictures/100002010000017600000108E352CABB877B9BBA.png
   :width: 3.2965in
   :height: 2.3272in
.. |image125| image:: Pictures/1000020100000176000001082C32F9B76F45EF0D.png
   :width: 3.2957in
   :height: 2.3264in
.. |image126| image:: Pictures/100002010000018000000108AB81D4EA0126F6F3.png
   :width: 3.2965in
   :height: 2.2661in
.. |image127| image:: Pictures/10000201000001760000010801E062E3B7559967.png
   :width: 3.2965in
   :height: 2.3272in
.. |image128| image:: Pictures/100002010000018400000108CBAAC82BE8BE453B.png
   :width: 3.2957in
   :height: 2.2425in
.. |image129| image:: Pictures/100002010000018400000108092F47F7E23EC231.png
   :width: 3.2965in
   :height: 2.2429in
.. |image130| image:: Pictures/100002010000017600000108CEC263FBF9951747.png
   :width: 3.2965in
   :height: 2.3272in
.. |image131| image:: Pictures/100002010000017600000108E430FCAEAF4D4EC1.png
   :width: 3.2957in
   :height: 2.3264in
.. |image132| image:: Pictures/100002010000018000000108DCDEBF6DE49A8039.png
   :width: 3.2965in
   :height: 2.2661in
.. |image133| image:: Pictures/1000020100000176000001083D1614429327B946.png
   :width: 3.2965in
   :height: 2.3272in
.. |image134| image:: Pictures/100002010000018400000108CEDE2FDB1838709E.png
   :width: 3.2957in
   :height: 2.2425in
.. |image135| image:: Pictures/100002010000018400000108D27011B569F9D0D1.png
   :width: 3.2965in
   :height: 2.2429in
.. |image136| image:: Pictures/100002010000017600000108A1679F78A429409A.png
   :width: 3.2965in
   :height: 2.3272in
.. |image137| image:: Pictures/1000020100000176000001080D906599B465DC56.png
   :width: 3.2957in
   :height: 2.3264in
.. |image138| image:: Pictures/10000201000001800000010807CC013BC365161F.png
   :width: 3.2965in
   :height: 2.2661in
.. |image139| image:: Pictures/100002010000017600000108F652A97E248CE6DC.png
   :width: 3.2965in
   :height: 2.3272in
.. |image140| image:: Pictures/1000020100000184000001085FD66AF92A555730.png
   :width: 3.2957in
   :height: 2.2425in
.. |image141| image:: Pictures/1000020100000184000001087FDDD3B858833ACB.png
   :width: 3.2965in
   :height: 2.2429in
.. |image142| image:: Pictures/100002010000017600000108259A29825BD4FE6E.png
   :width: 3.2965in
   :height: 2.3272in
.. |image143| image:: Pictures/100002010000017600000108071D8DC7831B00EC.png
   :width: 3.2957in
   :height: 2.3264in
.. |image144| image:: Pictures/100002010000018000000108EB5A61E19F38A796.png
   :width: 3.2965in
   :height: 2.2661in
.. |image145| image:: Pictures/100002010000017600000108E5893982C5D53614.png
   :width: 3.2965in
   :height: 2.3272in
.. |image146| image:: Pictures/1000020100000184000001085220E57F602D61B9.png
   :width: 3.2957in
   :height: 2.2425in
.. |image147| image:: Pictures/100002010000018400000108D887005A4D55D4E0.png
   :width: 3.2965in
   :height: 2.2429in
.. |image148| image:: Pictures/100002010000017600000108EA7DD554888B9CB0.png
   :width: 3.2965in
   :height: 2.3272in
.. |image149| image:: Pictures/1000020100000176000001086492FAAB3DF54568.png
   :width: 3.2957in
   :height: 2.3264in
.. |image150| image:: Pictures/100002010000018000000108BBA999F34E96963F.png
   :width: 3.2965in
   :height: 2.2661in
.. |image151| image:: Pictures/100002010000017600000108B8315E78D029E10E.png
   :width: 3.2965in
   :height: 2.3272in
.. |image152| image:: Pictures/100002010000018400000108E4DF87E12A831E6C.png
   :width: 3.2957in
   :height: 2.2425in
.. |image153| image:: Pictures/100002010000018400000108C6A93F9A472DC6DA.png
   :width: 3.2965in
   :height: 2.2429in
.. |image154| image:: Pictures/100002010000017600000108D72E50083F9E84C9.png
   :width: 3.2965in
   :height: 2.3272in
.. |image155| image:: Pictures/100002010000017600000108BC4E239A3D122A17.png
   :width: 3.2957in
   :height: 2.3264in
.. |image156| image:: Pictures/1000020100000180000001087D76A7AC76E14BD5.png
   :width: 3.2965in
   :height: 2.2661in
.. |image157| image:: Pictures/100002010000017600000108EDE913BBF73803F1.png
   :width: 3.2965in
   :height: 2.3272in
.. |image158| image:: Pictures/1000020100000184000001080C78CD5AD2D8EDDC.png
   :width: 3.2957in
   :height: 2.2425in
.. |image159| image:: Pictures/100002010000018400000108BAFC52125D9DD7FC.png
   :width: 3.2965in
   :height: 2.2429in
.. |image160| image:: Pictures/1000020100000176000001086D1D12C73ECC6CAC.png
   :width: 3.2965in
   :height: 2.3272in
.. |image161| image:: Pictures/1000020100000178000001085463950D1DB773F9.png
   :width: 3.2957in
   :height: 2.3138in
.. |image162| image:: Pictures/10000201000001770000010804DB7FE35442ABB6.png
   :width: 3.2965in
   :height: 2.3201in
.. |image163| image:: Pictures/100002010000018A00000108C25AC5752C8899D1.png
   :width: 3.2965in
   :height: 2.2083in
.. |image164| image:: Pictures/10000201000001780000010810922793CA911E7A.png
   :width: 3.2957in
   :height: 2.3138in
.. |image165| image:: Pictures/100002010000017700000108557E8B7D35B9BFA2.png
   :width: 3.2965in
   :height: 2.3201in
.. |image166| image:: Pictures/100002010000018A000001086A2F90EA3DBD7C08.png
   :width: 3.2965in
   :height: 2.2083in
.. |image167| image:: Pictures/100002010000017800000108563B8B02D337FD40.png
   :width: 3.2957in
   :height: 2.3138in
.. |image168| image:: Pictures/100002010000017700000108463836B73F1F2B70.png
   :width: 3.2965in
   :height: 2.3201in
.. |image169| image:: Pictures/100002010000018A00000108F718911EC03B5EBF.png
   :width: 3.2965in
   :height: 2.2083in
.. |image170| image:: Pictures/1000020100000178000001085187E9FE99B80002.png
   :width: 3.2957in
   :height: 2.3138in
.. |image171| image:: Pictures/1000020100000177000001084DFAD0DC11C77702.png
   :width: 3.2965in
   :height: 2.3201in
.. |image172| image:: Pictures/100002010000018A00000108C5B3E70B0340EE21.png
   :width: 3.2965in
   :height: 2.2083in
.. |image173| image:: Pictures/100002010000017800000108BB7D275CABBF763B.png
   :width: 3.2957in
   :height: 2.3138in
.. |image174| image:: Pictures/1000020100000177000001087765C4DD5DF97EAE.png
   :width: 3.2965in
   :height: 2.3201in
.. |image175| image:: Pictures/100002010000018A00000108740F5871889ED2A6.png
   :width: 3.2965in
   :height: 2.2083in
.. |image176| image:: Pictures/1000020100000178000001085D70E3C27EFE30B7.png
   :width: 3.2957in
   :height: 2.3138in
.. |image177| image:: Pictures/100002010000017700000108326B7DC120A1554D.png
   :width: 3.2965in
   :height: 2.3201in
.. |image178| image:: Pictures/100002010000018A00000108E3B2167646C82099.png
   :width: 3.2965in
   :height: 2.2083in
.. |image179| image:: Pictures/1000020100000178000001089276E6FF7D556AA5.png
   :width: 3.2957in
   :height: 2.3138in
.. |image180| image:: Pictures/100002010000017700000108684654A6E8742CA1.png
   :width: 3.2965in
   :height: 2.3201in
.. |image181| image:: Pictures/100002010000018A0000010809CD742F7E60D9A0.png
   :width: 3.2965in
   :height: 2.2083in
.. |image182| image:: Pictures/100002010000017900000108789AE1B972FC98DD.png
   :width: 3.2957in
   :height: 2.3075in
.. |image183| image:: Pictures/100002010000017700000108BFEF576CC8AE1ED6.png
   :width: 3.2965in
   :height: 2.3201in
.. |image184| image:: Pictures/100002010000018A000001085BCDC25AA99B2939.png
   :width: 3.2965in
   :height: 2.2083in
.. |image185| image:: Pictures/10000201000001790000010846E51DAB211829EB.png
   :width: 3.2957in
   :height: 2.3075in
.. |image186| image:: Pictures/100002010000017700000108F79C946A32CB7318.png
   :width: 3.2965in
   :height: 2.3201in
.. |image187| image:: Pictures/100002010000018A000001080D15C77A62412C54.png
   :width: 3.2965in
   :height: 2.2083in
.. |image188| image:: Pictures/100002010000017800000108D8268AA41D4CFBC3.png
   :width: 3.2957in
   :height: 2.3138in
.. |image189| image:: Pictures/1000020100000177000001083F95B7EBD045E937.png
   :width: 3.2965in
   :height: 2.3201in
.. |image190| image:: Pictures/100002010000018A00000108DDDF5838AE0B3CEC.png
   :width: 3.2965in
   :height: 2.2083in
.. |image191| image:: Pictures/100002010000017C0000010858C8E35C7FD20116.png
   :width: 3.2957in
   :height: 2.2898in
.. |image192| image:: Pictures/100002010000017B00000108285ABA3FFAF295D4.png
   :width: 3.2965in
   :height: 2.2957in
.. |image193| image:: Pictures/100002010000018E00000108F1B6E9A28014B2D1.png
   :width: 3.2965in
   :height: 2.1862in
.. |image194| image:: Pictures/100002010000017C000001086B5639D9705C1B66.png
   :width: 3.2957in
   :height: 2.2898in
.. |image195| image:: Pictures/100002010000017B000001083EBCE35047E276A6.png
   :width: 3.2965in
   :height: 2.2957in
.. |image196| image:: Pictures/100002010000018E0000010829CBC15642CDC860.png
   :width: 3.2965in
   :height: 2.1862in
.. |image197| image:: Pictures/100002010000017C00000108E873D6076FF31170.png
   :width: 3.2957in
   :height: 2.2898in
.. |image198| image:: Pictures/100002010000017B000001083CFA3109522A310B.png
   :width: 3.2965in
   :height: 2.2957in
.. |image199| image:: Pictures/100002010000018E000001088F5D89B4C7BDB783.png
   :width: 3.2965in
   :height: 2.1862in
.. |image200| image:: Pictures/100002010000017C0000010836BFD6FEB36BAF68.png
   :width: 3.2957in
   :height: 2.2898in
.. |image201| image:: Pictures/100002010000017B000001089A63FD44412DC8F6.png
   :width: 3.2965in
   :height: 2.2957in
.. |image202| image:: Pictures/100002010000018E00000108A5792BFAC5298AF0.png
   :width: 3.2965in
   :height: 2.1862in
.. |image203| image:: Pictures/100002010000017C0000010857BC47D30FF9A4B0.png
   :width: 3.2957in
   :height: 2.2898in
.. |image204| image:: Pictures/100002010000017B000001087BDA0EF6BA5CF729.png
   :width: 3.2965in
   :height: 2.2957in
.. |image205| image:: Pictures/100002010000018E00000108F910B04CA83808D4.png
   :width: 3.2965in
   :height: 2.1862in
.. |image206| image:: Pictures/100002010000017C000001085802DFEFD2D01F63.png
   :width: 3.2957in
   :height: 2.2898in
.. |image207| image:: Pictures/100002010000017B0000010878CA71EDD0B51C2A.png
   :width: 3.2965in
   :height: 2.2957in
.. |image208| image:: Pictures/100002010000018E000001083EF13F6D6AF03E91.png
   :width: 3.2965in
   :height: 2.1862in
.. |image209| image:: Pictures/100002010000017C0000010876D46B8D465101B4.png
   :width: 3.2957in
   :height: 2.2898in
.. |image210| image:: Pictures/100002010000017B0000010825EDAC5C7F8FA12A.png
   :width: 3.2965in
   :height: 2.2957in
.. |image211| image:: Pictures/100002010000018E00000108BEE39218E924631C.png
   :width: 3.2965in
   :height: 2.1862in
.. |image212| image:: Pictures/100002010000017C0000010858998BDF29A74294.png
   :width: 3.2957in
   :height: 2.2898in
.. |image213| image:: Pictures/100002010000017B000001084256FF19F7EE741B.png
   :width: 3.2965in
   :height: 2.2957in
.. |image214| image:: Pictures/100002010000018E0000010841B0821B6B79AE8E.png
   :width: 3.2965in
   :height: 2.1862in
.. |image215| image:: Pictures/100002010000017C00000108AC122DEE0F72A597.png
   :width: 3.2957in
   :height: 2.2898in
.. |image216| image:: Pictures/100002010000017B00000108D01643A91C7BE633.png
   :width: 3.2965in
   :height: 2.2957in
.. |image217| image:: Pictures/100002010000018E000001084A7CF6FCA91BEE80.png
   :width: 3.2965in
   :height: 2.1862in
.. |image218| image:: Pictures/100002010000017C00000108351302CEE7EA370A.png
   :width: 3.2957in
   :height: 2.2898in
.. |image219| image:: Pictures/100002010000017B00000108B74F8C3094D21960.png
   :width: 3.2965in
   :height: 2.2957in
.. |image220| image:: Pictures/100002010000018E0000010809730C15A0D91D23.png
   :width: 3.2965in
   :height: 2.1862in

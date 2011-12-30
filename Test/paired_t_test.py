import numpy as np
from scipy import stats

# http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

# Parameters :	
# a, b : sequence of ndarrays
# The arrays must have the same shape, except in the dimension corresponding to axis (the first, by default).
# axis : int, optional
# Axis can equal None (ravel array first), or an integer (the axis over which to operate on a and b).
# Returns :	
# t : float or array
# t-statistic
# prob : float or array
# two-tailed p-value
# Notes

# We can use this test, if we observe two independent samples from the same or different population, 
# e.g. exam scores of boys and girls or of two ethnic groups. The test measures whether the average (expected) 
# value differs significantly across samples. If we observe a large p-value, for example larger than 0.05 or 0.1, 
# then we cannot reject the null hypothesis of identical average scores. If the p-value is smaller than the threshold, 
# e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.

# Prob
a_list = [0.92518, 0.920813, 0.916187, 0.917937, 0.919389, 0.919693, 0.922979, 0.915242, 0.92143, 0.923757, 0.918867, 0.923973, 0.923241, 0.9122, 0.922315, 0.921498, 0.922163, 0.923086, 0.924178, 0.923008, 0.921065, 0.918661, 0.91762, 0.91877, 0.919336, 0.92287, 0.921103, 0.920099, 0.916902, 0.923219, 0.916339, 0.927525, 0.914411, 0.91971, 0.91324, 0.919802, 0.916929, 0.922102, 0.918217, 0.919906, 0.918617, 0.92471, 0.925692, 0.929116, 0.92037, 0.917538, 0.918783, 0.916637, 0.92214, 0.919408]
# Pred
b_list = [0.912444, 0.905142, 0.911231, 0.911966, 0.908979, 0.911558, 0.910017, 0.905232, 0.913816, 0.915513, 0.91389, 0.913803, 0.910411, 0.910329, 0.917092, 0.916316, 0.909359, 0.909894, 0.91104, 0.914397, 0.904975, 0.91008, 0.90823, 0.905914, 0.911103, 0.914082, 0.911288, 0.902057, 0.907013, 0.910762, 0.912236, 0.908513, 0.905031, 0.910932, 0.907846, 0.912403, 0.912898, 0.914154, 0.914087, 0.911307, 0.905149, 0.912087, 0.91602, 0.919419, 0.915241, 0.909664, 0.909901, 0.909997, 0.90863, 0.908933]
# Ridge
c_list = [0.910194, 0.90378, 0.909376, 0.911886, 0.909855, 0.911118, 0.911557, 0.908497, 0.913076, 0.914568, 0.908651, 0.91665, 0.907512, 0.906249, 0.909873, 0.916197, 0.91279, 0.90644, 0.910363, 0.915341, 0.910804, 0.911737, 0.906428, 0.907844, 0.906923, 0.910982, 0.909429, 0.901147, 0.906038, 0.909842, 0.907282, 0.907351, 0.90209, 0.907164, 0.909219, 0.913721, 0.910182, 0.910425, 0.915182, 0.912559, 0.902911, 0.911428, 0.915866, 0.917805, 0.915091, 0.903406, 0.914322, 0.907596, 0.906708, 0.908963]
# NB
d_list = [0.909979, 0.893379, 0.904901, 0.913678, 0.904431, 0.910748, 0.901927, 0.908186, 0.905041, 0.905395, 0.908436, 0.900476, 0.914032, 0.903658, 0.907489, 0.914856, 0.900633, 0.906148, 0.905182, 0.909052, 0.90676, 0.900235, 0.90162, 0.898749, 0.903604, 0.912888, 0.898172, 0.896642, 0.904965, 0.909505, 0.900711, 0.90289, 0.900705, 0.90883, 0.904673, 0.90357, 0.905869, 0.9087, 0.906153, 0.913031, 0.904501, 0.902279, 0.907812, 0.915127, 0.907171, 0.904501, 0.908248, 0.899316, 0.900496, 0.903064]



a = np.asarray(a_list)
b = np.asarray(b_list)
c = np.asarray(c_list)
d = np.asarray(d_list)

# More details: http://mathworld.wolfram.com/Pairedt-Test.html
# See http://www.scipy.org/doc/api_docs/SciPy.stats.stats.html
print 'ac'
print stats.ttest_ind(a,c,axis = None)
print stats.ttest_rel(a,c,axis = None)
print 'bc'
print stats.ttest_ind(b,c,axis = None)
print stats.ttest_rel(b,c,axis = None)
print 'dc'
print stats.ttest_ind(d,c,axis = None)
print stats.ttest_rel(d,c,axis = None)
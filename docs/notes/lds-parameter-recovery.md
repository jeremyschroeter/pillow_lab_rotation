# Procedure:
	1. Generate data from a LLDS
	2. Fit a an LLDS model with increasing amounts of data
		1. I fixed the maximum number of EM iterations to 100 (is this good?)
	3. Log the MSE between the fit and ground-truth LLDS parameters

One thing I am unsure about is I am re-initializing the models when I show it more data. Not sure if I should be starting from where I ended on the last fitting. I think it makes intuitive sense to do that, but I am not here.
### LDS model
![[Pasted image 20260323110732.png|500]]


I did the dame procedure with the CTDS model which inherits most functionality from the LDS model, but has different M steps
### CTDS Model
![[Pasted image 20260323105822.png|500]]


# Every subplot above should be decreasing monotonically, but fails to do so. The numbers are small which is good, but they aren't getting any smaller. Not sure if this is just convergence and deviations are numerical or if something is wrong.
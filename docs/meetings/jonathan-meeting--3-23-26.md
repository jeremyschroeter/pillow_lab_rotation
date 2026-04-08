# Parameter recovery
## Procedure:
	1. Generate data from a LLDS
	2. Fit a an LLDS model with increasing amounts of data
		1. I fixed the maximum number of EM iterations to 100 (is this good?)
	3. Log the MSE between the fit and ground-truth LLDS parameters


### LDS model
![[Pasted image 20260323110732.png|800]]


I did the dame procedure with the CTDS model which inherits most functionality from the LDS model, but has different M steps
### CTDS Model
![[Pasted image 20260323105822.png|800]]


# Every subplot above should be decreasing monotonically, but that's not what I see. The numbers are small which is good, but they aren't getting any smaller. Not sure if this is just convergence and deviations are numerical or if something is wrong.




Stop when the delta LL is below some criterion

Try doing the EM updates for individual parameters and see if the LL is still monotonically increasing with EM iterations
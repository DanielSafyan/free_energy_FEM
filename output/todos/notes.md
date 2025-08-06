# Notes



## Electrical Current Experiment observations
![alt text](currentelectrode1.png) 

this is the current of one electrode in an NPPwithFOReaction simulation. Stimuling electrode [nan, v*, nan, v*], Sensing Electrode [v*/100] 

- this does not look quite as expected
    -> I think the reason is that too little time is given, e.g. Diffusion works much slower than the time was given to the the system. There are orders of magnitude difference between the time scales of diffusion and the time scale of the experiment. But still you can see some kind of memory in the system being measured. 




## 3D 
- I think that for visualization it might be reasonable to project the 3D simulation to 2D by summing along the z axis, since the pong game is symmetric enough along this axis 
# Notes



## Electrical Current Experiment observations
![alt text](currentelectrode1.png) 

this is the current of one electrode in an NPPwithFOReaction simulation. Stimuling electrode [nan, v*, nan, v*], Sensing Electrode [v*/100] 

- this does not look quite as expected

    -> I think the reason is that too little time is given, e.g. Diffusion works much slower than the time was given to the the system. There are orders of magnitude difference between the time scales of diffusion and the time scale of the experiment. But still you can see some kind of memory in the system being measured. 

- **After running the experiment in 3 dimensions it looks very much like in the paper!!!**
![alt text](3Dcurrentelectrode1.png)

    -> Even though right now I have only run it for 10 time steps and 10x10x10 nodes

    -> Also I have added  another stimulating elctrode pair on the other side of the sensing electrodes: 
         ![alt text](ElectrodeConfiguration.png)

    -> The configuration does not seem to matter as much. I have tried again with a kind of pong game configuration without the top stimulating row and got the same graph as a result!!!!


- **This set of equation can simulate the hysterisis memory property of the hydrogel!!!**

    -> It actually would be more accurate to say its a property of the a nernst planck model, since all conditions are not specific to hydrogel

## 3D 
- I think that for visualization it might be reasonable to project the 3D simulation to 2D by summing along the z axis, since the pong game is symmetric enough along this axis 

    -> looks kinda weird, but may also be because we only have 10x10x10 nodes (cubic scaling is really unfortunate)

    -> 10x10x10 nodes for 10 steps took 2:18 minutes 

        -> 2x in nodes would 8x in time

        -> wish list: 20x20x20 nodes for 300 steps would take: 10 hours




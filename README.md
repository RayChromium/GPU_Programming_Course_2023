# ÅAU GPU Programming 2023 (IT00CG19-3002)  
  
FYI: I'm just trying to figure out how to solve the problem given from the lectures, here's just my thoughts on it, and I assume **it may not be the correct/only answer**.  

## Background  
The final project is to solve a problem given from the slides using cuda programming. Problem is here:    
![problem description 1](images/problem_description_1.png)    
![problem description 2](images/problem_description_2.png)    
  
Let's try to break this down step by step to understand what should we actually do.  
  
### Histograms DD, DR and RR for the two point angular  
  
First things first, we need 3 histograms ``DD``, ``DR`` and ``RR``.   
### What is a "Histogram"? 
According to [Wikipedia](https://en.wikipedia.org/wiki/Histogram), a **histogram** is a diagram is an approximate representation of the distribution of numerical data, which look like this:  
![example histogram](images/Example_histogram.png)  
  
Let's say there are 100 entries of data ( in this case integers representing arivals ) in the collection. Each rectengle represents the frequency of data within a range, for example the first rectengle at the right side of 2 on the x/horizontal axis reprecents **the numbers of arrivals between 2 and 2.5 minute**, and all the heights of the rectangles sum up to be 100.  
Here's some terminologies:  
1. The width (on the x/horizontal) of each rectangle is called bin width, marked with ``h`` , then the  number of bins is marked with ``k``:  
![bins width](images/bins_width_k.svg)  
2.  If we let ``n`` be the total number of observations and ``k`` be the total number of bins, the height of rectangle ``i``  (labled with ``m_i``) meet the following conditions:  
![Alt text](images/n_and_heights_mi.svg)  

#### How to calculate ``DD``, ``RR`` and ``DR``
  
In our case, the rectangle ``DD`` stores frequncies of angles. Each on of theses angle is the angle between 2 vectors, aka 2 different galaxies given the **real measured galaxies** data source. These vectors are given in spherical coordinates, each of them has 2 components, according to the slides: 
![input data 1](images/input_data_1.png)  
![input data 2](images/input_data_2.png)  
  
And for each 2 of the vectors we can get the cosine of their angle if we divide the **dot product** by **the product of their length**:  
![coord to angle 1](images/coord_to_cos_1.png)  
![coord to angle 2](images/coord_to_cos_2.png)  
![coord to angle 3](images/coord_to_cos_3.png)  
  
Therefore the cuda function might look like this:  

### Normalization and calculating the eveness between R and D   
  


### Visualizing  
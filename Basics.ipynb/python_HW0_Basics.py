#!/usr/bin/env python
# coding: utf-8

# <h2>Project 0: Python Basics</h2>

# <h3>Introduction</h3>
# 
# <p>In this project, you will write a function to compute Euclidean distances between sets of vectors, and get familiar with the Vocareum system that we will be using this semester. </p>
# 
# <p><strong>Academic Integrity:</strong> We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; <em>please</em> don't let us down. If you do, we will pursue the strongest consequences available to us.
# 
# <p><strong>Getting Help:</strong> You are not alone!  If you find yourself stuck  on something, contact the course staff for help.  Office hours, section, and the <a href="https://edstem.org/us/courses/8451/discussion/">Ed Discussion</a> are there for your support; please use them.  If you can't make our office hours, let us know and we will schedule more.  We want these projects to be rewarding and instructional, not frustrating and demoralizing.  But, we don't know when or how to help unless you ask.

# # Numpy
# In this and future projects should you choose you use python, you will make a great deal of use of the numpy package. Numpy is a package that contains many routines for fast matrix and vector operations. Behind the scenes, rather than executing slow Python code, numpy functions often execute code that is compiled and highly optimized.
# 
# If you are not familiar with the Numpy package, you can read an overview of it <a href="https://numpy.org/doc/stable/user/quickstart.html/">here</a>, and find a full API <a href="https://docs.scipy.org/doc/numpy/reference/">here</a>. We import numpy for you below. Also, as a check, your Python version should be 3.x (for some value of x). 

# In[1]:


#<GRADED>
import sys
import numpy as np # Numpy is Python's built in library for matrix operations.
                   # We will be using it a lot in this class!
#<GRADED>
from pylab import * 
print('You\'re running python %s' % sys.version.split(' ')[0])


# 
# <h3> Euclidean distances in Python </h3>
# 
# <p>Many machine learning algorithms access their input data primarily through pairwise (Euclidean) distances. It is therefore important that we have a fast function that computes pairwise distances of input vectors. </p>
# <p>Assume we have $n$ data vectors $\vec x_1,\dots,\vec x_n\in{\cal R}^d$ and $m$ vectors $\vec z_1,\dots,z_m\in{\cal R}^d$. With these data vectors, let us define two matrices $X=[\vec x_1,\dots,\vec x_n]\in{\cal R}^{n\times d}$, where the $i^{th}$ row is a vector $\vec x_i$ and similarly $Z=[\vec z_1,\dots,\vec z_m]\in{\cal R}^{m\times d}$. </p>
# <p>We want a distance function that takes as input these two matrices $X$ and $Z$ and outputs a matrix $D\in{\cal R}^{n\times m}$, where 
# 	$$D_{ij}=\sqrt{(\vec x_i-\vec z_j)(\vec x_i-\vec z_j)^\top}.$$
# </p>

# A naïve implementation to compute pairwise distances may look like the code below:

# In[2]:


def l2distanceSlow(X,Z=None):
    if Z is None:
        Z = X
    
    n, d = X.shape     # dimension of X
    m= Z.shape[0]   # dimension of Z
    D=np.zeros((n,m)) # allocate memory for the output matrix
    for i in range(n):     # loop over vectors in X
        for j in range(m): # loop over vectors in Z
            D[i,j]=0.0; 
            for k in range(d): # loop over dimensions
                D[i,j]=D[i,j]+(X[i,k]-Z[j,k])**2; # compute l2-distance between the ith and jth vector
            D[i,j]=np.sqrt(D[i,j]); # take square root
    return D


# Please read through the code carefully and make sure you understand it. It is perfectly correct and will produce the correct result ... eventually. To see what is wrong, try running the l2distanceSlow code on an extremely small matrix X:
# 

# In[3]:


X=np.random.rand(700,100)
print("Running the naive version for the first time ...")
get_ipython().run_line_magic('time', 'Dslow=l2distanceSlow(X)')


# This code defines some random data in $X$ and computes the corresponding distance matrix $D$. The <em>%time</em> statements time how long this takes. When I ran the code, the <em>l2distanceSlow</em> function took <strong>43.6s to run</strong>! 
# 
# This is an appallingly large amount of time for such a simple operation on a small amount of data, and writing code like this to deal with matrices in this class will result in code that takes <strong>days</strong> to run. 
# 
# 
# <strong>As a general rule, you should avoid tight loops at all cost.</strong> As we will see in the remainder of this assignment, we can do much better by performing bulk matrix operations using the <em>numpy</em> package, which calls highly optimized compiled code behind the scenes.
# 
# 
# 
# 

# <h4> How to program in NumPy </h4>
# 
# <p>Although there is an execution overhead per line in Python, matrix operations are extremely optimized and very fast. In order to successfully program in this course, you need to free yourself from "for-loop" thinking and start thinking in terms of matrix operations. Python for scientific computing can be very fast if almost all the time is spent in a few heavy duty matrix operations. In this assignment you will do this, and transform the function above into a few matrix operations <em>without any loops at all.</em> </p> 
# 
# <p>The key to efficient programming in Python for machine learning in general is to think about it in terms of mathematics, and not in terms of Loops. </p>
# 
# <p>	(a) Show that the Gram matrix (aka inner-product matrix)
# $$	G_{ij}=\mathbf{x}_i\mathbf{z}_j^\top $$
# can be expressed in terms of a pure matrix multiplication. Once you are done with the derivation, implement the function <strong><code>innerproduct</code></strong>.</p>
# 

# __a) TODO: Please fill in your answer in this Markdown cell (double-click to edit)__
# 
#             your answer goes here...

# In[4]:


def innerproduct(X,Z=None):
    # function innerproduct(X,Z)
    #
    # Computes the inner-product matrix.
    # Syntax:
    # D=innerproduct(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix G of size nxm
    # G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
    #
    # call with only one input:
    # innerproduct(X)=innerproduct(X,X)
    #
        
    # your code goes here ...
    raise NotImplementedError('Your code goes here!')
    # until here 


# If your code is correct you should pass the following two tests below. 

# In[5]:


# a simple test for the innerproduct function
M=np.array([[1,2,3],[4,5,6],[7,8,9]])
Q=np.array([[11,12,13],[14,15,16]])
assert (all(diag(innerproduct(M))==[14,77,194])) # test1: Inner product with itself
print("You passed test#1\n")
assert (np.all(innerproduct(M,Q).T==np.array([[74,182,290],[92,227,362]])))
print("You passed test#2\n")


# (b) Let us define two new matrices $S,R\in{\cal R}^{n\times m}$ 
# 		$$S_{ij}=\mathbf{x}_i\mathbf{x}_i^\top, \ \ R_{ij}=\mathbf{z}_j\mathbf{z}_j^\top.$$
#  	Show that the <em>squared</em>-euclidean matrix $D^2\in{\cal R}^{n\times m}$, defined as
# 		$$D^2_{ij}=(\mathbf{x}_i-\mathbf{z}_j)(\mathbf{x}_i-\mathbf{z}_j)^\top,$$
# 	can be expressed as a linear combination of the matrix $S, G, R$. (Hint: It might help to first express $D^2_{ij}$ in terms of inner-products.) What do you need to do to obtain the true Euclidean distance matrix $D$?</p></td>
# <p>
#     
# Think about what the $S$ and $R$ matrices look like. You will find that the values in each row of $S$ and the values in each column of $R$ do not change! This is also apparent when considering that $S_{ij} = \mathbf{x}_i \mathbf{x}_i^\top$ for all $j$ ; similar argument for $R_{ij} = \mathbf{z}_j \mathbf{z}_j^\top$ for all $i$.
# $$
# S = \begin{bmatrix}
# \mathbf{x}_1 \mathbf{x}_1^\top & \mathbf{x}_1 \mathbf{x}_1^\top & \cdots & \mathbf{x}_1 \mathbf{x}_1^\top\\
# \mathbf{x}_2 \mathbf{x}_2^\top & \mathbf{x}_2 \mathbf{x}_2^\top & \cdots & \mathbf{x}_2 \mathbf{x}_2^\top\\
# \vdots & \vdots & \ddots & \vdots\\
# \mathbf{x}_n \mathbf{x}_n^\top & \mathbf{x}_n \mathbf{x}_n^\top & \cdots & \mathbf{x}_n \mathbf{x}_n^\top\\
# \end{bmatrix}, \ 
# R = \begin{bmatrix}
# \mathbf{z}_1 \mathbf{z}_1^\top & \mathbf{z}_2 \mathbf{z}_2^\top & \cdots & \mathbf{z}_m \mathbf{z}_m^\top\\
# \mathbf{z}_1 \mathbf{z}_1^\top & \mathbf{z}_2 \mathbf{z}_2^\top & \cdots & \mathbf{z}_m \mathbf{z}_m^\top\\
# \vdots & \vdots & \ddots & \vdots\\
# \mathbf{z}_1 \mathbf{z}_1^\top & \mathbf{z}_2 \mathbf{z}_2^\top & \cdots & \mathbf{z}_m \mathbf{z}_m^\top\\
# \end{bmatrix}.
# $$
# 
# For more information on the shape of S, let's take a look at the definition of $S$: 
# 
# $$
# S = \begin{bmatrix}
# \mathbf{x}_1 \mathbf{x}_1^\top & \mathbf{x}_1 \mathbf{x}_1^\top & \cdots & \mathbf{x}_1 \mathbf{x}_1^\top\\
# \mathbf{x}_2 \mathbf{x}_2^\top & \mathbf{x}_2 \mathbf{x}_2^\top & \cdots & \mathbf{x}_2 \mathbf{x}_2^\top\\
# \vdots & \vdots & \ddots & \vdots\\
# \mathbf{x}_n \mathbf{x}_n^\top & \mathbf{x}_n \mathbf{x}_n^\top & \cdots & \mathbf{x}_n \mathbf{x}_n^\top\\
# \end{bmatrix}
# $$
# 
# Here, $x_n x_n^T$ does not mean the $n$th row, $n$th column. They are just the names of the vectors from $X$ - but not indices of the matrix $S$. You could just as easily call $x_1$ the vector $1_x$, $2_x$ the vector $x_2$, etc...to make it clear that they aren't indices, and rewrite the matrix that way. (Confusing right!?) What $S$ means is, this is the column vector $(x_1x_1^T, \ldots, x_n x_n^T)^T$ (which is $n$ numbers long) copied horizontally $m$ times. Therefore, $S$ is actually an $n \times m$ matrix.
# 

# __b) TODO:Please fill in your answer in this Markdown cell (double-click to edit)__
# 
#             your answer goes here...

# <p>	(c) Implement the function <strong><code>l2distance</code></strong>, which computes the Euclidean distance matrix $D$ without a single loop:
#    </p>
# <p><strong>Hint</strong>: Make sure that when you take the square root of the squared distance matrix, ensure that all entries are non-negative. Sometimes very small positive numbers can become negative due to numerical imprecision. Knowing that all distances must always be non-negative, you can simply overwrite all negative values as 0.0 to avoid unintended consequences </p>

# In[6]:


#<GRADED>
def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #

    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    # Your code goes here ..
    raise NotImplementedError('Your code goes here!')
    # ... until here
#</GRADED>


# In[7]:


# Little test of the distance function
X1=rand(2,3);
print("The diagonal should be (more or less) all-zeros:", diag(l2distance(X1,X1)))
assert(all(diag(l2distance(X1,X1))<=1e-7))
print("You passed l2distance test #1.")

X2=rand(5,3);
Dslow=l2distanceSlow(X1,X2);
Dfast=l2distance(X1,X2);
print("The norm difference between the distance matrices should be very close to zero:",norm(Dslow-Dfast))
assert(norm(Dslow-Dfast)<1e-7)
print("You passed test #2.")

x1=np.array([[0,1]])
x2=np.array([[1,0]])
x1.shape
x2.shape
print("This distance between [0,1] and [1,0] should be about sqrt(2): ",l2distance(x1,x2)[0,0])
assert(norm(l2distance(x1,x2)[0,0]-sqrt(2))<1e-8)
print("You passed l2distance test #3.")


# Let us compare the speed of your l2-distance function against our previous naïve implementation:

# In[8]:


import time
current_time = lambda: int(round(time.time() * 1000))

X=np.random.rand(700,100)
Z=np.random.rand(300,100)

print("Running the naïve version ...")
before = current_time()
Dslow=l2distanceSlow(X)
after = current_time()
t_slow = after - before
print("{:2.2f}s".format(t_slow))

print("Running the vectorized version ...")
before = current_time()
Dfast=l2distance(X)
after = current_time()
t_fast = after - before
print("{:2.2f}s".format(t_fast))


speedup = t_slow / t_fast
print("The two method should deviate by very little {:05.6f}".format(norm(Dfast-Dslow)))
print("but your numpy code was {:05.2f} times faster!".format(speedup))


# How much faster is your code now? With this implementation you should easily be able to compute the distances between <strong>many more</strong> vectors. You can easily see how, even for small datasets, the for-loop based implementation could take several days or even weeks to perform basic operations that take seconds or minutes with well-written numpy code.

# In[ ]:





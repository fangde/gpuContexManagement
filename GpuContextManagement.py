
# coding: utf-8

# In[2]:


import tensorflow as tf


import pycuda.driver as cuda

from pycuda.compiler import SourceModule


import numpy as np
import uuid 
# In[5]:

ImageDB = {}


#creat the contex
cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()


ctx.pop()

def PutImage(vol):

    
    name = str(uuid.uuid4())

    volc = vol.astype(np.float32)
    volc = np.ascontiguousarray(volc)
    ctx.push()
    cuda_array_in = cuda.np_to_array(volc, order='C')
    ctx.pop()
    ImageDB.update({name: cuda_array_in})

    return name




if __name__=="__main__":

    sess=tf.Session()
    x=tf.placeholder(tf.float32,shape=[32,32,1])
    y=tf.placeholder(tf.float32,shape=[32,32,1])

    
    sum=x+y

    for i in range(5):
        data=np.zeros(shape=(128,128,32),dtype=np.float32)
        hash=PutImage(data)

        dx=np.ones(shape=(32,32,1),dtype=np.float32) 
        dy=np.zeros(shape=(32,32,1),dtype=np.float32)


        dsum=sess.run(sum,feed_dict={x:dx,y:dy})   
        print dsum.shape,hash


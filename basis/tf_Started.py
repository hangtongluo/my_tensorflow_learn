#Getting Started With TensorFlow
import tensorflow as tf

#create two floating point Tensors node1 and node2
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)
print(node1, node2)

#Notice that printing the nodes does not output the values 3.0 and 4.0 as you might expect

#To actually evaluate the nodes, we must run the computational graph within a session. 
#A session encapsulates the control and state of the TensorFlow runtime.

sess = tf.Session()
print('values:',sess.run([node1, node2]))

#add our two constant nodes and produce a new graph
node3 = tf.add(node1, node2)
print('node3:', node3)
print('sess.run(node3):', sess.run(node3))




import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
import numpy as np

if __name__ == "__main__":
    # Create a dummy data for linear regression
    x_data = np.arange(10,step=0.01)
    y_data = 2*x_data+10#+np.sin(x_data/10)
    #plt.scatter(x_data,y_data)
    #plt.show()

    nsamples = 1000
    nbatch  = 100
    X_data = np.reshape(x_data,(nsamples,1))
    Y_data = np.reshape(y_data,(nsamples,1))

    # Creating placeholders for input
    X = tf.placeholder(tf.float32,shape=(nbatch,1))
    y = tf.placeholder(tf.float32,shape=(nbatch,1))
    sess = tf.InteractiveSession()
    # Create everything within a variable scope
    with tf.variable_scope("linear-regression"):
        W = tf.get_variable("weights",(1,1),initializer =\
                tf.random_normal_initializer())
        b = tf.get_variable("bias",(1,),initializer = \
                tf.constant_initializer(3.0))
        y_pred = tf.add(tf.matmul(X,W),b)
        tf.histogram_summary('pred',y_pred)
        loss = tf.reduce_sum((y-y_pred)**2/nbatch)
        # Create an optimizer
        opt = tf.train.AdamOptimizer().minimize(loss)
        tf.scalar_summary('loss',loss)
        tf.scalar_summary('weight',W[0,0])
        tf.scalar_summary('bias',b[0])
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter('train/',
                                      sess.graph)

        sess.run(tf.initialize_all_variables())
        print "Initial slope: ",W[0,0].eval(), " and bias: ",b[0].eval()
        print "Training, this might take a while ..."
        for i in range(30000):
            indices = np.random.choice(nsamples,nbatch)
            X_batch,y_batch = X_data[indices],Y_data[indices]
            # Do gradient descent
            _,loss_val,summary = sess.run([opt,loss,merged],\
                    feed_dict = {X:X_batch,y:y_batch})
            train_writer.add_summary(summary,i)
        print "Resulting slope: ",W[0,0].eval(), " and bias: ",b[0].eval()

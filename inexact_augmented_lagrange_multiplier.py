import numpy as np 
from numpy.linalg import norm, svd
import tensorflow as tf
from keras.backend import clear_session

def inexact_augmented_lagrange_multiplier(X, lmbda = 0.01, tol = 1e-8, maxIter = 1000):
   
    Y = X
    # norm_two = norm(np.sign(Y.ravel()), 2)
    # norm_inf = norm(np.sign(Y.ravel()), np.inf) / lmbda#无穷范数
    # Y = np.sign(Y) /dual_norm
    norm_two=norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda#无穷范数
    dual_norm = np.max([norm_two, norm_inf])
    Y=Y/dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n= Y.shape[1]
    itr = 0
    aa=True
    while True:
        Eraw = X - A + (1/mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        usv=X - Eupdate + (1 / mu) * Y
        U, S, V = svd(usv, full_matrices=False)#A=USV.T(81920, 100) (100,) (100, 100)
        # print(U.shape, S.shape, V.shape)
        svp = (S > 1 / mu).shape[0]
        svp=100
        # if svp < sv:
            # sv = np.min([svp + 1, n])
        # else:
            # sv = np.min([svp + round(0.05 * n), n])

        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        mu_1=1 / mu
        # print( np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)).shape,V[:svp, :].shape)
        # print(S[:svp] - 1 / mu)
        A = Aupdate
        E = Eupdate
        #print itr
        Z = X - A - E
        Y = Y + mu * Z
        
        mu = np.min([mu * rho, mu * 1e7])
        if mu * rho< mu * 1e7:
            aa=False
        else:
            aa=True
        itr += 1
        
        step=(norm(Z, 'fro') / dnorm)
        print('iteration:',itr,'. error:',step,'mu',mu,aa,'mean:',np.mean(A),np.mean(E),np.mean(Y))
        if ( step< tol) or (itr >= maxIter): 
            break
    print("IALM Finished at iteration %d" % (itr))
    return A, E


def inexact_augmented_lagrange_multiplier_tf(X, lmbda = 0.01, tol = 1e-9, maxIter = 1000):
    clear_session()
    shape=X.shape
    print(shape)
    svp=100
    input_1=X.copy()
    Y_value = X
    # norm_two = norm(np.sign(Y.ravel()), 2)
    # norm_inf = norm(np.sign(Y.ravel()), np.inf) / lmbda#无穷范数
    # Y = np.sign(Y) /dual_norm
    norm_two=norm(Y_value.ravel(), 2)
    norm_inf = norm(Y_value.ravel(), np.inf) / lmbda#无穷范数
    dual_norm = np.max([norm_two, norm_inf])
    Y_value=Y_value/dual_norm
    # A = np.zeros(Y.shape)
    # E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')

    D=tf.placeholder(tf.float64,[shape[0],shape[1]],name='input1')
    A=tf.placeholder(tf.float64,[shape[0],shape[1]])
    E=tf.placeholder(tf.float64,[shape[0],shape[1]])
    Y=tf.placeholder(tf.float64,[shape[0],shape[1]])
    mu=tf.placeholder(tf.float64)
    
    dnorm=tf.cast(tf.norm(D),tf.float64)
    rho=tf.constant(1.5,tf.float64)
    lmbda=tf.constant(0.01,tf.float64)
    
    mu_1=tf.cast(tf.divide(1,mu),tf.float64)
    Eraw = D - A + tf.multiply(mu_1, Y) # no porblem
    
    min=tf.cast(tf.divide(lmbda,mu),tf.float64)
    Eupdate = tf.maximum(Eraw - min , 0.0) + tf.minimum(Eraw + min, 0.0)
    
    USV = D - Eupdate + mu_1 * Y
    S, U, V = tf.svd(USV, full_matrices=False)  # no porblem
    # print(S, U, V)
    V=tf.transpose(V,perm=[1,0])
    Aupdate = tf.matmul(tf.matmul(U[:, :svp], tf.diag(S[:svp] - tf.cast(mu_1,tf.float64))), V[:svp, :])
    Z = D - Aupdate - Eupdate
    Yupdate = Y + mu*Z
    mu_c = tf.minimum(mu * rho, mu * 1e7)
    Z_norm=tf.norm(Z)
    step = tf.divide(Z_norm,dnorm)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    i=0
    A_init=np.zeros(input_1.shape)
    E_init=np.zeros(input_1.shape)
    Y_init=Y_value
    mu_init=1.25 / norm_two
    A_value,E_value,Y_value,mu_value,error=sess.run([Aupdate,Eupdate,Yupdate,mu_c,step],feed_dict={D:input_1,A:A_init,E:E_init,Y:Y_init,mu:mu_init})
    while True:#
        # A_value,E_value,dnorm_=sess.run([Z_norm,step,dnorm],feed_dict={D:input_1})
        # print('dnorm:',dnorm_)
        # sess.run(feed_dict={D:input_1,A:A_value,E:E_value,Y:Y_value,mu:mu_value})
        
        A_value,E_value,Y_value,mu_value,error=sess.run([Aupdate,Eupdate,Yupdate,mu_c,step],feed_dict={D:input_1,A:A_value,E:E_value,Y:Y_value,mu:mu_value})
        print('iteration:',i,'error:',error,'mu_value:',mu_value)
        i=i+1
        if error<tol:
            break
    return A_value,E_value
    
    
    
   
def inexact_augmented_lagrange_multiplier_tf_sgd(X, lmbda = 0.01, tol = 1e-9, maxIter = 1000):
    clear_session()
    shape=X.shape
    print(shape)
    svp=100
    input_1=X.copy()
    Y_value = X
    # norm_two = norm(np.sign(Y.ravel()), 2)
    # norm_inf = norm(np.sign(Y.ravel()), np.inf) / lmbda#无穷范数
    # Y = np.sign(Y) /dual_norm
    norm_two=norm(Y_value.ravel(), 2)
    norm_inf = norm(Y_value.ravel(), np.inf) / lmbda#无穷范数
    dual_norm = np.max([norm_two, norm_inf])
    Y_value=Y_value/dual_norm
    # A = np.zeros(Y.shape)
    # E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')

    D=tf.placeholder(tf.float64,[shape[0],shape[1]],name='input1')
    A=tf.placeholder(tf.float64,[shape[0],shape[1]])
    E=tf.placeholder(tf.float64,[shape[0],shape[1]])
    Y=tf.placeholder(tf.float64,[shape[0],shape[1]])
    mu=tf.placeholder(tf.float64)
    
    dnorm=tf.cast(tf.norm(D),tf.float64)
    rho=tf.constant(1.5,tf.float64)
    lmbda=tf.constant(0.01,tf.float64)
    
    mu_1=tf.cast(tf.divide(1,mu),tf.float64)
    Eraw = D - A + tf.multiply(mu_1, Y) # no porblem
    
    min=tf.cast(tf.divide(lmbda,mu),tf.float64)
    Eupdate = tf.maximum(Eraw - min , 0.0) + tf.minimum(Eraw + min, 0.0)
    
    USV = D - Eupdate + mu_1 * Y
    S, U, V = tf.svd(USV, full_matrices=False)  # no porblem
    print(S, U, V)
    V=tf.transpose(V,perm=[1,0])
    # U, S, V= tf.svd(USV, full_matrices=False)
    print('aaaaaaaaaaaaaaa',tf.shape(S[:svp] - tf.cast(mu_1,tf.float64)))
    Aupdate = tf.matmul(tf.matmul(U[:, :svp], tf.diag(S[:svp] - tf.cast(mu_1,tf.float64))), 
                                V[:svp, :])
    # Aupdate = tf.matmul(tf.matmul(U, tf.diag_part(S - tf.cast(mu_1,tf.float64))), V)
    check_Eraw=tf.reduce_mean(Aupdate)
    # A = Aupdate
    # E = Eupdate
    Z = D - Aupdate - Eupdate
    Yupdate = Y + mu*Z
    # mu_c =mu * rho
    mu_c = tf.minimum(mu * rho, mu * 1e7)
    Z_norm=tf.norm(Z)
    step = tf.divide(Z_norm,dnorm)
    
    # with tf.Graph().as_default():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    i=0
    A_init=np.zeros(input_1.shape)
    E_init=np.zeros(input_1.shape)
    Y_init=Y_value
    mu_init=1.25 / norm_two
    check_Eraw_1,A_value,E_value,Y_value,mu_value,error=sess.run([check_Eraw,Aupdate,Eupdate,Yupdate,mu_c,step],feed_dict={D:input_1,A:A_init,E:E_init,Y:Y_init,mu:mu_init})
    print(check_Eraw_1.shape)
    while True:#
        # A_value,E_value,dnorm_=sess.run([Z_norm,step,dnorm],feed_dict={D:input_1})
        # print('dnorm:',dnorm_)
        # sess.run(feed_dict={D:input_1,A:A_value,E:E_value,Y:Y_value,mu:mu_value})
        
        check_Eraw_1,A_value,E_value,Y_value,mu_value,error=sess.run([check_Eraw,Aupdate,Eupdate,Yupdate,mu_c,step],feed_dict={D:input_1,A:A_value,E:E_value,Y:Y_value,mu:mu_value})
        print('iteration:',i,'error:',error,'mu_value:',mu_value,check_Eraw_1)
        i=i+1
        if error<tol:
            break
    return A_value,E_value

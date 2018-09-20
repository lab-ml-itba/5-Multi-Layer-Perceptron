from sklearn.linear_model import LogisticRegression
import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
from keras.layers.advanced_activations import LeakyReLU

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def get_polynimial_set(X, degree = 12, bias = True):
    # Recibe el dataset X de numero_de_muestras x features  y devuelve una matriz con todas las combinaciones 
    # De los productos del grado indicado en degree
    k = 2
    n = degree + k
    pos = 0
    X_mat = np.zeros((X.shape[0],nCr(n,k)))
    for i in range(degree + 1):
        for j in range(i+1):
            X_mat[:,pos] = (X[:,0]**(i-j))*X[:,1]**j
            pos = pos + 1
    if bias:
        return X_mat
    else:
        return X_mat[:,1:]

def plot_boundaries_SVM(X_train, y_train, score=None, class_func=None, support_vectors=None,degree = None, n_colors = 100, mesh_res = 1000, ax = None):
    X = X_train #np.vstack((X_test, X_train))
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if class_func is not None:
        if degree is not None:
            polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree)
            Z = class_func(polynomial_set)[:, 1]
        else:
            Z_aux = class_func(np.c_[xx.ravel(), yy.ravel()])
            Z = Z_aux[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
    
        cf = ax.contourf(xx, yy, Z, n_colors, vmin=0., vmax=1., cmap=cm, alpha=.8)
        plt.colorbar(cf, ax=ax)
        #plt.colorbar(Z,ax=ax)

        boundary_line = np.where(np.abs(Z-0.5)<0.001)

        ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=40, horizontalalignment='right')

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100, marker='o')
    ax.scatter(support_vectors[:,0], support_vectors[:, 1], c='y',
               edgecolors='k', s=30, marker='o')
    
def plot_boundaries(X_train, y_train, score=None, probability_func=None, degree = None, n_colors = 100, mesh_res = 1000, ax = None):
    X = X_train #np.vstack((X_test, X_train))
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if probability_func is not None:
        if degree is not None:
            polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree)
            Z = probability_func(polynomial_set)[:, 1]
        else:
            Z_aux = probability_func(np.c_[xx.ravel(), yy.ravel()])
            Z = Z_aux[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
    
        cf = ax.contourf(xx, yy, Z, n_colors, vmin=0., vmax=1., cmap=cm, alpha=.8)
        plt.colorbar(cf, ax=ax)
        #plt.colorbar(Z,ax=ax)

        boundary_line = np.where(np.abs(Z-0.5)<0.001)

        ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=40, horizontalalignment='right')

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100, marker='o')
    

def fit_and_get_regions(X_train, y_train, X_test, y_test, degree = 2, lambd = 0, plot_it = True, print_it = False):
    X_train_degree = get_polynimial_set(X_train, degree=degree)
    X_test_degree = get_polynimial_set(X_test, degree=degree)
    # Defino el modelo de clasificación como Regresion Logistica
    if lambd == 0:
        C1 = 10000000000
    else:
        C1 = 1/lambd 
    #C2 = 1
    clf_logist_pol = LogisticRegression(C=C1, fit_intercept=False)

    # Entreno el modelo con el dataset de entrenamiento
    clf_logist_pol.fit(X_train_degree, y_train)

    # Calculo el score (Exactitud) con el dataset de testeo
    score_test_logist_pol = clf_logist_pol.score(X_test_degree, y_test)

    # Calculo tambien el score del dataset de entrenamiento para comparar
    score_train_logist_pol = clf_logist_pol.score(X_train_degree, y_train)
    
    #loss_train = _logistic_loss(clf_logist_pol.coef_, X_train_degree, y_train, 1 / clf_logist_pol.C)
    #loss_test = _logistic_loss(clf_logist_pol.coef_, X_test_degree, y_test, 1 / clf_logist_pol.C)

    # print('Test Accuracy (Exactitud):',score_test_logist_pol)
    # print('Train Accuracy (Exactitud):',score_train_logist_pol)
    # print('coeficientes:', clf_logist_pol.coef_)
    # print('intercept:', clf_logist_pol.intercept_)
    if plot_it:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
        plot_boundaries(X_train, y_train, score_train_logist_pol, clf_logist_pol.predict_proba, degree=degree, ax=ax1)
        plot_boundaries(X_test, y_test, score_test_logist_pol, clf_logist_pol.predict_proba, degree=degree, ax=ax2)
        print('Regresion Logistica Polinomial de orden '+str(degree) +', con lamdba (regularización L2):' +  str(lambd))
        plt.show()
    if print_it:
        print('Train Accuracy (Exactitud):',score_train_logist_pol)
        print('Test Accuracy (Exactitud):',score_test_logist_pol)
    return score_train_logist_pol, score_test_logist_pol, clf_logist_pol.coef_ #, loss_train, loss_test

def test_options(X_train, y_train, X_test, y_test, options, plot_it=False):
    train_acc_array = []
    test_acc_array = []
    degrees = []
    lambdas = []
    coefs_array_mean = []
    coefs_array_std = []
    coefs_abs_max = []
    coefs_norm = []
    coefs_num = []
    for opt in options:
        tr_acc, ts_acc, coefs = fit_and_get_regions(X_train, y_train, X_test, y_test, degree = opt['degree'], lambd = opt['lambd'], plot_it=plot_it)
        train_acc_array.append(tr_acc)
        test_acc_array.append(ts_acc)
        degrees.append(opt['degree'])
        lambdas.append(opt['lambd'])
        coefs_num.append(coefs.shape[1])
        coefs_array_mean.append(coefs.mean())
        coefs_array_std.append(coefs.std())
        coefs_abs_max.append(np.max(abs(coefs)))
        coefs_norm.append(np.linalg.norm(coefs))
    return degrees, lambdas, train_acc_array, test_acc_array, coefs_array_mean, coefs_array_std, coefs_abs_max, coefs_norm, coefs_num

def plot_boundaries_keras(X_train, y_train, score, probability_func, model_vals=None,degree=None, bias=False, h = .02, ax = None, margin=0.5, mesh_res = 500,activation='sigmoid'):
    colors=['g','w','y','b']
    X = X_train
    margin_x = (X[:, 0].max() - X[:, 0].min())*margin
    margin_y = (X[:, 1].max() - X[:, 1].min())*margin
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree, bias=bias)
        Zaux = probability_func(polynomial_set)
    else:
        Zaux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z_aux[:, 1]
    print(Zaux.shape)
    
    if Zaux.shape[1] == 2:
        Z = Zaux[:, 1]
    else:
        Z = Zaux[:, 0]
        
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    cf = ax.contourf(xx, yy, Z, 50, cmap=cm, alpha=.8)
    plt.colorbar(cf, ax=ax)
    #plt.colorbar(Z,ax=ax)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k', s=100)
    if activation=='tanh':
        CT=0
    elif activation=='sigmoid':
        CT=0.5
    elif activation=='relu':
        CT=0.011
    for num,model_val in enumerate(model_vals):
        Z_th=list()
        Zaux=model_val.predict(np.c_[xx.ravel(), yy.ravel()])
        for idx in range(Zaux.shape[1]):
            Z_th.append(Zaux[:,idx].reshape(xx.shape))
        for z_th in Z_th:
            boundary_line = np.where(np.abs(z_th-CT)<0.01)
            ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color=colors[num], alpha=0.5, s=10)
    boundary_line = np.where(np.abs(Z-0.5)<0.01)
    ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')
    
from keras import regularizers
from keras.initializers import RandomUniform
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras import optimizers
from fnn_helper import PlotLosses
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint,TensorBoard

def get_n_layer_model_L2(input_shape, output_size, hidden_units=None,dropout=0.0, lr=0.1, l2_lambda=0, decay=0.0, initializer="normal", l1_lambda=0, optim = None,activation='tanh',tensorboard=False,logdir="./logs"):
    model = Sequential()
    #if type(hidden_units==int):
    #    hidden_units=[hidden_units]
    if tensorboard:
        callbackTB=TensorBoard(log_dir=logdir, histogram_freq=1000, batch_size=32, write_graph=True, write_grads=True, write_images=True)
    if optim is None:
        optim = optimizers.adam(lr=lr, decay=decay)
    regularizer = regularizers.l2(0)
    if (l2_lambda > 0):
        regularizer = regularizers.l2(l2_lambda)
    if (l1_lambda > 0):
        regularizer = regularizers.l1(l1_lambda)
    if hidden_units:
        model.add(Dense(hidden_units[0],
                        input_dim=input_shape,  
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        activation=activation,
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       ))
        model.add(Dropout(dropout))
        for layer_hidden_units in hidden_units[1:]:
            model.add(Dense(layer_hidden_units,  
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        activation=activation,
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       ))
            model.add(Dropout(dropout))
        model.add(Dense(output_size, 
                        activation='sigmoid', 
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        name='Salida',
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       ))
        model.compile(loss = 'binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    else:
        model.add(Dense(output_size,input_dim=input_shape,  
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        activation='sigmoid', 
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       ))
        model.compile(loss = 'binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    if tensorboard:
        return model, callbackTB
    else:
        return model, None
def get_two_layer_model_L2(input_shape, output_size, hidden_units=10, lr=0.1, l2_lambda=0, decay=0.0, initializer='normal', l1_lambda=0, optim = None):
    model = Sequential()
    if optim is None:
        optim = optimizers.adam(lr=lr, decay=decay)
    regularizer = regularizers.l2(0)
    if (l2_lambda > 0):
        regularizer = regularizers.l2(l2_lambda)
    if (l1_lambda > 0):
        regularizer = regularizers.l1(l1_lambda)
    model.add(Dense(hidden_units,input_dim=input_shape,  
                    kernel_initializer=initializer, 
                    bias_initializer=initializer,
                    activation='sigmoid', 
                    kernel_regularizer=regularizer, 
                    bias_regularizer=regularizer
                   ))
    model.add(Dense(output_size, 
                    activation='sigmoid', 
                    kernel_initializer=initializer, 
                    bias_initializer=initializer,
                    name='Salida',
                    kernel_regularizer=regularizer, 
                    bias_regularizer=regularizer
                   ))
    model.compile(loss = 'binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

def get_basic_model(input_shape, output_size, lr=0.1):
    model = Sequential()
    # optim = optimizers.SGD(lr=lr)
    optim = optimizers.adam(lr=lr)
    model.add(Dense(output_size, input_dim=input_shape,
                    activation='sigmoid', 
                    kernel_initializer='normal', 
                    name='Salida'
                   ))
    model.compile(loss = 'binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

def plot_th_MLP(hidden_units,activation='tanh', optim = optimizers.adam(lr=0.01, decay=0.0001),batch_size = 32, epochs = 10000, dropout=0.0, l2_lambda=0, initializer="normal", l1_lambda=0):
    import numpy as np
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy') 
    y_test = np.load('y_test.npy')
    input_shape = 2
    output_size = 1
    plot_losses = PlotLosses(plot_interval=200, evaluate_interval=None, x_val=X_test, y_val_categorical=y_test)
    if optim is None:
        optim = optimizers.adam(lr=lr, decay=decay)
    regularizer = regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)
    model_vals=list()
    if hidden_units:
        input_model = Input(shape=(2,))
        h1=Dense(hidden_units[0],
                        input_dim=input_shape,  
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        activation=activation,
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       )(input_model)
        model_vals.append(Model(inputs=[input_model], outputs=[h1]))
        d=Dropout(dropout)(h1)
        for layer_hidden_units in hidden_units[1:]:
            h=Dense(layer_hidden_units,  
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        activation=activation,
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       )(d)
            model_vals.append(Model(inputs=[input_model], outputs=[h]))
            d=Dropout(dropout)(h)
        h=Dense(output_size, 
                        activation='sigmoid', 
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        name='Salida',
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       )(d)
        model = Model(inputs=[input_model], outputs=[h])
        model.compile(loss = 'binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    else:
        input_model = Input(shape=(2,))
        h1=Dense(hidden_units[0],
                        input_dim=input_shape,  
                        kernel_initializer=initializer, 
                        bias_initializer=initializer,
                        activation=activation,
                        kernel_regularizer=regularizer, 
                        bias_regularizer=regularizer
                       )(input_model)
        model = Model(inputs=[input_model], outputs=[h1])
        model_val= Model(inputs=[input_model], outputs=[h1])
        model.compile(loss = 'binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    plot_losses = PlotLosses(plot_interval=200, evaluate_interval=None, x_val=X_test, y_val_categorical=y_test)
    model.fit(X_train, 
          y_train, batch_size = batch_size,
          epochs=epochs, 
          verbose=0, 
          validation_data=(X_test, y_test), 
          callbacks=[plot_losses],
         )
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
    plot_boundaries_keras(X_train, y_train, model.evaluate(X_train, y_train)[1], model.predict, model_vals, h = 0.01, margin=0.1, ax=ax1, activation=activation)
    plot_boundaries_keras(X_test, y_test, model.evaluate(X_test, y_test)[1], model.predict,model_vals, h = 0.01, margin=0.1, ax=ax2,  activation=activation)
    plt.show()
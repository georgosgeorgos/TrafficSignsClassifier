def labels_distro(y_train):
    
    labels = {i:0 for i in set(y_train)}
    
    for i in y_train:
        labels[i] += 1
        
    labels = {i:labels[i]*100/len(y_train) for i in labels}
    
    return labels


def worker(data):
    
    '''
    :for processes
    '''
    
    sess = tf.Session()
    (X, y, ix) = data
    
    length = len(y)
    images = []
    labels = []
    
    start = ix * length

    for i in range(0, length):
        
        if distro[y[i]] < 1:

            img = X[i]
            img = np.reshape(img, (32,32,1))
            r = np.random.random()
            noise = np.random.normal(size=[32,32,1])

            if r < 0.25:
                img = sess.run(tf.image.flip_left_right(img))

            elif r < 0.50:
                img = sess.run(tf.image.flip_up_down(img))

            elif r < 0.75:
                img = sess.run(tf.image.transpose_image(img))

            elif r < 1:
                img = sess.run(tf.image.rot90(img))


            img = img + noise
            images.append(img)
            labels.append(y[i])

            if i % 100 == 0:
                print(start + i)
                
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


def generate_batch_pool(X_train, y_train, n_pool):

    data = []
    length_tot = len(y_train)
    offset = math.ceil(len(y_train)/n_pool)

    for i in range(n_pool):

        start = int( i * offset )
        end = int( (i+1) * offset )
        data.append((X_train[start:end], y_train[start:end], i))
        
    return data


def parallel(X_train, y_train, worker, n_pool=4):
    
    data = generate_batch_pool(X_train, y_train, n_pool)
    
    pool = ThreadPool(n_pool)
    results = pool.map(worker, data)
    pool.close() 
    pool.join()
    
    return results

def concatenate(results, X_train, y_train):

    for i in range(len(results)):

        images = results[i][0]
        labels = results[i][1]
        print(i)
        X_train = np.vstack((X_train, images))
        y_train = np.hstack((y_train, labels))
    
    return X_train, y_train
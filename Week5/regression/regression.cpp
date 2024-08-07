#include "regression.h"

Regression::Regression(uint64_t D){
    d = D;
    weights = zeros(d,1);
    weights_ = zeros(d,1);
    bias = 0;
    epsilon = EPS;
    eta = ETA;
}

matrix Regression::transform(matrix X){
    uint64_t n = X.shape().first;
    matrix PHI(n,d);
    // Compute the transform of X as defined in README.md
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < d; ++j) {
            PHI(i, j) = pow(X(i, 0), j + 1);
        }
    }
    return PHI;
}

double Regression::l2loss(matrix X, matrix Y){
    matrix Y_pred = matmul(X,weights) + bias;
    __size d1 = Y.shape(), d2 = Y_pred.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot compute loss of vectors with dimensions ( "+to_string(d1.first)+" , "
        +to_string(d1.second)+" ) and ( "+to_string(d2.first)+" , "+to_string(d2.second)+" ) do not match");
    }
    double loss = 0;
    // Compute the mean squared loss as defined in README.md
    for (uint64_t i = 0; i < X.shape().first; i++) {
        loss += pow(Y_pred(i) - Y(i), 2);
    }
    loss /= X.shape().first;
    return loss;
}

pair<matrix, double> Regression::l2lossDerivative(matrix X, matrix Y){
    matrix Y_pred = matmul(X,weights) + bias;
    __size d1 = Y.shape(), d2 = Y_pred.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot compute loss derivative of vectors with dimensions ( "+to_string(d1.first)+" , "
        +to_string(d1.second)+" ) and ( "+to_string(d2.first)+" , "+to_string(d2.second)+" ) do not match");
    }
    //Compute gradients as defined in README.md
    matrix dw(d,1);
    double db;
    dw = matmul(X.transpose(), (Y_pred - Y));
    dw = dw / X.shape().first;
    for (uint64_t i = 0; i < X.shape().first; i++) {
        db += Y_pred(i) - Y(i);
    }
    db /= X.shape().first;
    return {dw,db};
}

matrix Regression::predict(matrix X){
    matrix Y_pred(X.shape().first,1);
    // Using the weights and bias, find the values of y for every x in X
    Y_pred = matmul(X, weights) + bias;
    return Y_pred;
}  

void Regression::GD(matrix X, matrix Y,double learning_rate, uint64_t limit){
    eta = learning_rate;
    double old_loss = 0,loss = l2loss(X,Y);
    train_loss.PB(loss);
    uint64_t iteration = 0;
    max_iterations = limit;
    while (fabs(loss - old_loss) > epsilon && iteration < max_iterations){
        // Calculate the gradients and update the weights and bias correctly. Do not edit anything else
        pair<matrix, double> gradient = l2lossDerivative(X, Y);
        weights -= eta * gradient.first;
        bias -= eta * gradient.second;
        old_loss = loss;
        loss = l2loss(X, Y);
        if (iteration %100 == 0) train_loss.PB(loss);
        iteration++;
    }
}

void Regression::train(matrix X,matrix Y,double learning_rate, uint64_t limit){

    /* 
     * These lines are for computing the closed form solution of weights in the case of zero bias. 
     * This is merely a reference and you do not need to try and match the predictions from the trained weights with 
     * those from weight_. 
     * However you do need to get a good enough accuracy and that is obtained by varying the learning rate and 
     * maximum number of iterations .
    */
    matrix PHI = transform(X);
    matrix PHIt = PHI.transpose();
    matrix Z = matmul(PHIt,PHI);
    GD(PHI,Y,learning_rate,limit);
    weights_ = matmul((Z.inverse()),matmul(PHIt,Y));
    cout << "Training Loss\n";
    for (uint64_t i = 0; i < train_loss.size() ; i++){
        cout << train_loss[i] << "\n";
    }
}

void Regression::test(matrix X,matrix Y){
    matrix PHI = transform(X);
    matrix Y_pred = predict(PHI);
    uint64_t n = PHI.shape().first;
    matrix Y_closed = matmul(PHI,weights_);
    cout << "Predictions(GD) \t Predictions(C) \t True Value\n"; 
    for (uint64_t i = 0 ; i < n ; i++){
        cout << Y_pred(i,0) << "\t\t\t "<< Y_closed(i,0) << "\t\t\t " << Y(i,0) << "\n";
    }
    cout << "Testing loss " << l2loss(PHI,Y) << "\n";
    cout << "Testing accuracy " << accuracy(Y_pred,Y) << "\n";

}

double Regression::accuracy(matrix Y_pred, matrix Y){
    double acc = 0;
    // Compute the accuracy of the model
    for (uint64_t i = 0; i < Y_pred.shape().first; i++) {
        acc += fabs(Y_pred(i) - Y(i)) / fabs(Y(i));
    }
    acc /= Y_pred.shape().first;
    return acc;
}

pair<pair<matrix, matrix>, pair<matrix, matrix>> test_train_split(matrix X, matrix Y, float ratio) {
    if (ratio < 0 || ratio > 1) {
        throw std::invalid_argument("Ratio must be between 0 and 1");
    }

    uint64_t n = X.shape().first;
    uint64_t train_size = static_cast<uint64_t>(n * ratio);
    uint64_t test_size = n - train_size;

    std::vector<uint64_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    matrix X_train(train_size, X.shape().second);
    matrix Y_train(train_size, 1);
    matrix X_test(test_size, X.shape().second);
    matrix Y_test(test_size, 1);

    for (uint64_t i = 0; i < train_size; ++i) {
        for (uint64_t j = 0; j < X.shape().second; ++j) {
            X_train(i, j) = X(indices[i], j);
        }
        Y_train(i, 0) = Y(indices[i], 0);
    }

    for (uint64_t i = 0; i < test_size; ++i) {
        for (uint64_t j = 0; j < X.shape().second; ++j) {
            X_test(i, j) = X(indices[train_size + i], j);
        }
        Y_test(i, 0) = Y(indices[train_size + i], 0);
    }

    return {{X_train, Y_train}, {X_test, Y_test}};
}
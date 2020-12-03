import numpy as np
import csv
import matplotlib.pylab as plt

# X=[0,1,1,3]
# 숫자로 주어지는 y값을 길이 vector_length인 one-hot 벡터로 변환합니다.
def convert_y_to_one_hot_vector(y,vector_length):
    y_vect=np.zeros((len(y),vector_length))
    for i in range(len(y)):
        y_vect[i,y[i]]=1
    return y_vect
# print(convert_y_to_one_hot_vector(X,4))

# 학습 데이터 세트 개수에서 라벨과 신경망 결과가 일치하지 않는 경우를 빼서 정확성 계산
def compute_accuracy(y_test,y_pred):
    size = y_test.shape[0]
    count = 0
    for i in range(size):
        diff=abs(np.argmax(y_test[i,:])-np.argmax(y_pred[i,:]))
        # print(diff)
        if diff !=0:
            count+=1
    return 100-count*100.0/size

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivation(x):
    return sigmoid(x) * (1-sigmoid(x))

def feedforward(x,W1,W2,b1,b2):
    # input layer
    a1=x
    # hidden layer
    z2=np.dot(W1,a1)+b1 #내적
    a2=sigmoid(z2)
    #output layer
    z3=np.dot(W2,a2)+b2
    a3=sigmoid(z3)

    return a1,a2,a3,z2,z3

def train(X,Y,node_size,max_iteration,learning_rate):
    # 초기 가중치 값으로 무작위 값을 사용
    # random.random(a,b) : a X b 행렬
    W2=np.random.random((node_size['output_layer_size'],node_size['hidden_layer_size']))
    W1=np.random.random((node_size['hidden_layer_size'],node_size['input_layer_size']))
    b2=np.random.random(node_size['output_layer_size'])
    b1=np.random.random(node_size['hidden_layer_size'])

    dataset_size=len(Y)
    list_avgerage_cost=[]
    list_accuracy=[]
    count=0

    while count < max_iteration:
        # 비어있는 넘파이 배열 사용
        dW2=np.zeros((node_size['output_layer_size'],node_size['hidden_layer_size']))
        dW1=np.zeros((node_size['hidden_layer_size'],node_size['input_layer_size']))
        db2=np.zeros(node_size['output_layer_size'])
        db1=np.zeros(node_size['hidden_layer_size'])

        average_cost=0

        # 학습 데이터 세트의 모든 샘플을 대상으로 피드포워드와 역전파 알고리즘을 수행
        for x,y in zip(X,Y):
            # 피드포워드
            a1,a2,a3,z2,z3=feedforward(x,W1,W2, b1, b2)
            # 역전파 알고리즘
            output_layer_error=y-a3
            delta3 = -(output_layer_error)*sigmoid_derivation(z3)
            average_cost+=np.linalg.norm((output_layer_error),2)/dataset_size

            hidden_layer_error = np.dot(W2.T,delta3)
            delta2 = hidden_layer_error*sigmoid_derivation(z2)

            dW2+=np.dot(delta3[:,np.newaxis],np.transpose(a2[:,np.newaxis]))/dataset_size
            db2+=delta3/dataset_size

            dW1+=np.dot(delta2[:,np.newaxis],np.transpose(a1[:,np.newaxis]))/dataset_size
            db1+=delta2/dataset_size

        # 역전파 알고리즘 실행 결과를 사용하여 신경망의 가중치와 편향을 업데이트
        W2+=-learning_rate*dW2
        b2+=-learning_rate*db2
        W1+=-learning_rate*dW1
        b1+=-learning_rate*db1

        # 예측해보고 정확도를 측정
        y_pred = predict_y(X,W1,W2,b1,b2)
        accuracy = compute_accuracy(Y,y_pred)

        # 매 반복 시 측정된 비용을 리스트에 저장
        list_accuracy.append(accuracy)
        list_avgerage_cost.append(average_cost)

        # 100번 반복시 비용 출력. (실행 시 비용이 감소하는 추이를 보는 데 사용)
        if count%100==0:
            print ('{}/{} cost : {}, Prediction accuracy : {}%'.format(count,max_iteration,average_cost,accuracy))
        count+=1
    return W1,W2,b1,b2,list_avgerage_cost,list_accuracy

# 주어진 테스트 데이터 세트와 가중치, 편향을 사용하여 신경망의 출력을 리턴
def predict_y(X,W1,W2,b1,b2):
    dataset_size=X.shape[0]
    y=np.zeros((dataset_size,3))
    for i in range(dataset_size):
        a1,a2,a3,z2,z3 = feedforward(X[i,:],W1,W2,b1,b2)
        y[i]=a3
    return y

if __name__=="__main__":
    # csv 하일로부터 데이터를 가져와서 가공

    # 붓꽃 품종을 딕셔너리로 정의하여 문자열로된 라벨값을 숫자값 라벨로 변환하는데 사용
    Species_Dict = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

    # 특성 (X)와 라벨 (Y)정의하는데 사용
    X=[]
    Y=[]

    # csv파일을 열어 한줄씩 가져옴
    with open('Iris.csv',newline='') as file:
        reader=csv.reader(file)
        try:
            for i, row in enumerate(reader):
                if i>0:
                    # csv로 부터 읽어온 데이터를 특성과 라벨로 나누어 리스트를 저장.
                    X.append(np.array(row[1:5],dtype="float64"))
                    # print(np.array(row[1:5]))
                    # 앞에서 정의한 딕셔너리를 이용해 문자열 라벨을 숫자 라벨로 변환
                    Y.append(Species_Dict[row[-1]])
            # 데이터가 저장된 리스트를 넘파이 배열로 변환
            X=np.array(X)
            Y=np.array(Y)
        except csv.Error as e:
            sys.exit('file {}, line {} : {}'.format(filename,reader.line_num,e))
    Y=convert_y_to_one_hot_vector(Y,vector_length=3)    #라벨 {0,1,2}를 one-hot 인코딩하여 {0 0 1, 0 1 0, 1 0 0} 으로 변환
    # print(Y)

    # 데이터 세트를 무작위로 섞음
    # print(Y.shape[0])
    s=np.arange(Y.shape[0]) #row 갯수 길이의 배열 생성
    # print(s)
    np.random.seed(0)
    np.random.shuffle(s)
    # print(s)
    Y=Y[s]
    X=X[s]

    # 학습용 데이터 (X_train, Y_train)와 테스트용 데이터 (X_test,Y_test)를 8:2 비율로 사용
    size=len(Y)
    p=int(size*0.8)

    X_train=X[0:p]
    Y_train=Y[0:p]
    X_test=X[p:]
    Y_test=Y[p:]
    # print(X_train[0])
    # print(Y_train[0])

    # 신경망 구조 정의
    node_size={
        'input_layer_size' : 4,
        'hidden_layer_size' : 8,
        'output_layer_size' : 3
    }

    # 역전파 알고리즘에 사용하는 학습률 정의
    learning_rate = 0.5

    # 신경망 학습 시작
    W1,W2,b1,b2,list_avg_cost,list_accuracy=train(
        X_train,Y_train,node_size=node_size,max_iteration=1000,learning_rate=learning_rate
    )

    Figure, ax = plt.subplots(1,2)
    ax[0].title.set_text('Average cost')
    ax[1].title.set_text('accuracy')

    ax[0].plot(list_avg_cost)
    ax[1].plot(list_accuracy)

    ax[0].set_xlabel('iteration number')
    ax[0].set_ylabel('Average cost')

    ax[1].set_xlabel('iteration number')
    ax[1].set_ylabel('accuracy')

    plt.show()

    y_pred = predict_y(X_test,W1,W2,b1,b2)
    print('Prediction accuracy : {}%'.format(compute_accuracy(Y_test,y_pred)))

import numpy as np
import matplotlib.pylab as plt

# <<<sigmoid function>>>
def draw_sigmoid():
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    x=np.arange(-10,10,0.1) #start, end, setp
    # print(x)
    W=np.arange(0.5,3,0.5)
    f=1/(1+np.exp(-x))

    ax.plot(x,f)
    ax.set_xticks(range(-10,10))
    ax.set_yticks(range(0,2))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()

def draw_sigmoid_with_W():
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    x=np.arange(-10,10,0.1) #start, end, setp
    # print(x)
    W=np.arange(0.5,3,0.5)
    # 입력 x 에 가중치 w 곱합
    for w in W:
        f=1/(1+np.exp(-x*w))
        ax.plot(x,f)
    ax.set_xticks(range(-10,10))
    ax.set_yticks(range(0,2))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()

def draw_sigmoid_with_bias():
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    x=np.arange(-10,10,0.1) #start, end, setp
    # print(x)
    w=0.5   #가중치 고정
    B=np.arange(-2,2,0.5)   #편향에 변화줌
    for b in B:
        # 입력 x에 가중치 w를 곱한 후 편향 b를 더합니다.
        f=1/(1+np.exp(-x*w+b))
        ax.plot(x,f)
    ax.set_xticks(range(-10,10))
    ax.set_yticks(range(0,2))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()

# <<<feedforward : 신경망의 출력을 계산하는 과정>>>
def sigmoid(x):
    return 1/(1+np.exp(-x))
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

# 신경망을 구성하는 레이어의 노드 개수 지정
# 신경망은 총 3개의 레이어로 구성
# 입력 레이어 노드 수 : 3, 히든 레이어 개수 : 3, 출력레이어 개수 1
node_size={
    'input_layer_size' : 3,
    'hidden_layer_size' : 3,
    'output_layer_size' : 1
}
# 초기 가중치 값으로 무작위 값을 사용
# random.random(a,b) : a X b 행렬
W2=np.random.random((node_size['output_layer_size'],node_size['hidden_layer_size']))
W1=np.random.random((node_size['hidden_layer_size'],node_size['input_layer_size']))
b2=np.random.random(node_size['output_layer_size'])
b1=np.random.random(node_size['hidden_layer_size'])

# 학습 데이터 세트
# X : 특성
# Y : 라벨
X=np.array([[1,0,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],[0,1,0],[1,1,1]])
Y=np.array([1,0,0,0,1,1,0])

# 특성 하나인 x에 대해 피드포워드를 수행
# 라벨 하나인 y는 cost 계산을 위해 사용
for x,y in zip(X,Y):
    # 특성과 가중치를 사용하여 피드포워드를 수행하고 결과를 리턴받음
    a1,a2,a3,z2,z3=feedforward(x,W1,W2,b1,b2)
print('a3={},y={},Error(L2Norm) ={}'.format(a3,y,np.linalg.norm(y-a3),2))   #L@ NORM 계산(비용 계산)


# 최솟값을 구할 2차함수
def f(x):
    return np.power(x-5,2)-20
# 주어진 2차함수의 도함수
def f_derivative(x):
    return 2*x-10
# 경사하강법을 구현한 함수
def gradient_descent(next_x,gamma,precision,max_iteration):
    # 반복할 때마다 이동한 거리의 변화 추이를 보기위해 리스트에 저장
    list_step=[]
    for i in range(max_iteration):  #주어진 함수의 최솟값을 찾기 위해 최대 max_iteration만큼 반복함
        current_x=next_x    #계산된 위치는 다음 번 반복시 현재위치로 사용
        next_x=current_x-gamma*f_derivative(current_x)  #현재위치에서 기울기를 뺀 위치를 update
        # 현재위치에서 다음 위치까지 이동하는 거리를 측정하여 리스트에 저장(x좌표 거리 기준)
        step = next_x-current_x
        list_step.append(abs(step))

        # 50번 반복마다 로그 출력
        if i%50==0:
            print('{}/{},x={:5.6f},'.format(i,max_iteration,current_x),end="")
            gradient=gamma*f_derivative(current_x)
            print('f(x)={:5.6f},'.format(f(current_x),gradient),end="")
            print('gradient sign={}'.format('+'if f_derivative(current_x)>0 else '-'))
        if abs(step)<=precision:    #지정한 값보다 이동한 거리가 작아지면 루프 중지
            break
    print('function has min when x={}.'.format(current_x))
    Figure,ax=plt.subplots(1,1)
    ax.title.set_text('step size')
    ax.plot(list_step)
    ax.set_ylabel('step size')
    ax.set_xlabel('Iteration number')
    plt.show()

# <<<역전파 알고리즘>>>
# 역전파 알고리즘 사용 시 활성화 함수의 1차 도함수가 필요
def sigmoid_derivation(x):
    return sigmoid(x) * (1-sigmoid(x))
learning_rate=2.0
count = 0   #반복 횟수를 카운트
max_iteration = 1000    #학습 데이터 세트 전체에 대한 피드포워드와 역전파를 1000번 반복
dataset_size = len(Y)   #학습 데이터 세트에 포함된 데이터의 갯수
list_average_cost=[]    #반복할 때마다 변하는 비용을 저장하기 위한 리스트
while count < max_iteration:
    # 역전파 알고리즘 적용 시 각 샘플별로 측정되는 값을 저장하기 위해 사용
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
        delta3 = -(y-a3)*sigmoid_derivation(z3)
        average_cost+=np.linalg.norm((y-a3),2)/dataset_size
        delta2 = np.dot(W2.T,delta3)*sigmoid_derivation(z2)
        dW2+=np.dot(delta3[:,np.newaxis],np.transpose(a2[:,np.newaxis]))/dataset_size
        db2+=delta3/dataset_size
        dW1+=np.dot(delta2[:,np.newaxis],np.transpose(a1[:,np.newaxis]))/dataset_size
        db1+=delta2/dataset_size

    # 역전파 알고리즘 실행 결과를 사용하여 신경망의 가중치와 편향을 업데이트
    W2+=-learning_rate*dW2
    b2+=-learning_rate*db2
    W1+=-learning_rate*dW1
    b1+=-learning_rate*db1

    # 매 반복 시 측정된 비용을 리스트에 저장
    list_average_cost.append(average_cost)

    # 100번 반복시 비용 출력. (실행 시 비용이 감소하는 추이를 보는 데 사용)
    if count%100==0:
        print ('{}/{} cost : {}'.format(count,max_iteration,average_cost))
    count+=1

    # 반복 횟수에 대비 비용 그래프를 그린다.
Figure, ax = plt.subplots(1,1)
ax.title.set_text('Average cost')
ax.plot(list_average_cost)
ax.set_xlabel('iteration number')
ax.set_ylabel('Average cost')
plt.show()

for x,y in zip(X,Y):
    # 피드포워드 실행
    a1,a2,a3,z2,z3 = feedforward(x,W1, W2, b1, b2)
    print(y)
    print(a3)

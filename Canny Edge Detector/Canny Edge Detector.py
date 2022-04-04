from collections import deque
from inspect import stack
import queue
from PIL import Image
import math
import numpy as np


def gauss1d(sigma):
    rounded_sigma = round(sigma*6) #sigma에 6을 곱한 값을 올림
    if(rounded_sigma%2 == 0): #올림값이 짝수면 1을 더하여 홀수로 만듬
        rounded_sigma += 1
    temp_array = np.arange(-(rounded_sigma//2), rounded_sigma//2+1) #0이 중앙값인 rounded_sigma크기의 배열을 만듬
    gauss_array = np.array([density_fun(x, sigma) for x in temp_array]) #각 배열에 density function 적용
    normalize_array = gauss_array/np.linalg.norm(gauss_array,1) #배열의 총합이 1이되게 정규화
    return normalize_array

def density_fun(x, sigma): 
    return math.exp((-x**2)/(2*(sigma**2)))

def gauss2d(sigma):
    gauss2d_array = np.outer(gauss1d(sigma),gauss1d(sigma).transpose()) #gauss1d array와 그것의 전치행렬을 외적하여 gauss2d array를 구한다.
    return gauss2d_array

def convolve2d(array, filter):
    array  = array.astype(np.float32)  #array 자료형을 float32로 바꿈
    filter = filter.astype(np.float32)  #필터 자료형을 float32로 바꿈
    filter = np.flip(filter) #convolution을 위해서 필터를 상하좌우 반전시킨다
    result=[] 
    ax,ay = np.shape(array)
    fx,fy = np.shape(filter)   #필터와 array의 x,y좌표 크기를 구함

    m = (fx-1)//2
    padding_array = np.pad(array ,((m,m),(m,m)), 'constant', constant_values = 0) #필터 사이즈에따라 array를 알맞은 사이즈로 패딩한다.

    for i in range(ax): 
        for j in range(ay): 
            result.append((padding_array[i:i+fx,j:j+fy]*filter).sum())  #for문을 돌며 픽셀마다 filter를 적용하고 그값은 result리스트에 저장한다.

    result=np.array(result).reshape(ax,ay)  #result 리스트를 알맞은 사이즈의 배열로 만들어준다.

    return result

def gaussconvolve2d(array,sigma):
    return convolve2d(array,gauss2d(sigma)) #array에 sigma값을 인자로준 gauss2d필터를 적용시킨다.

def sobel_filters(img):
    
    x_filter = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    y_filter = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    Ix = convolve2d(img,x_filter) 
    Iy = convolve2d(img,y_filter) #이미지 배열에 각 필터를 적용 시켜 Ix, Iy를 얻는다.
   
    G = np.hypot(Ix,Iy)
    theta = np.arctan(Iy/Ix)  #hypot과 arctan메소드를 이용해 G와 theta를 구한다.

    
    G = G / G.max() * 255 #G배열을 비율에 맞게 바꾼다.
    G = G.astype('uint8')  #Iy를 uint형 배열로 바꾼다.
    G = np.where(G<=0,0,G)
    G = np.where(G>=255,255,G) #각 채널의 화소마다 0과 255사이 값이 아닌 값을 제거한다.

    img_G = Image.fromarray(G)
    img_G.save('gx_igu.png','PNG')  #바꾼배열을 다시 이미지화 시켜 png파일로 저장한다.
    
    return (G, theta)

def non_max_suppression(G, theta):
    h,w = G.shape #G의 hight와 width를 구한다.
    n_m_s = np.zeros((h,w), dtype = np.int32) #non max suppression을 실행한후 배열을 저장할 배열이다.

    angle = theta * 180 / np.pi #라디안을 각도로 바꾼다.
    angle = np.where(angle<0,angle+180,angle) #-90도에서 90도 사이 값을 0도에서 180도 사이값으로 바꾼다.


    #제일 바깥 패딩된 픽셀을 제외하고 각 pixel을 탐색한다.
    for i in range(1, h - 1):
      for j in range(1, w - 1):
      
        #비교한 주변 pixel값을 가질 변수들
        neighbor_1 = 0
        neighbor_2 = 0
        
        #0도 혹은 180도에 가까운 theta를 가진 pixel은 위아래pixel을 neighbor에 저장한다.
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
          neighbor_1 = G[i, j + 1]
          neighbor_2 = G[i, j - 1]
        #45도에 가까운 theta값을 가진 pixel의 주변 pixel값을 neighbor에 저장한다.
        elif (22.5 <= angle[i, j] < 67.5):
          neighbor_1 = G[i + 1, j - 1]
          neighbor_2 = G[i - 1, j + 1]
        #90도에 가까운 theta값을 가진 pixel의 주변 pixel값을 neighbor에 저장한다.
        elif (67.5 <= angle[i, j] < 112.5):
          neighbor_1 = G[i + 1, j]
          neighbor_2 = G[i - 1, j]
        #135도에 가까운 theta값을 가진 pixel의 주변 pixel값을 neighbor에 저장한다.
        elif (112.5 <= angle[i, j] < 157.5):
          neighbor_1 = G[i - 1, j - 1]
          neighbor_2 = G[i + 1, j + 1]
        #저장된 neighbor들의 값보다 큰 pixel만 남기고 나머지는 0의 값으로 만든다.
        if (G[i, j] >= neighbor_1) and (G[i, j] >= neighbor_2):
          n_m_s[i, j] = G[i, j]
        
    img_nms = Image.fromarray(n_m_s.astype('uint8'))
    img_nms.save('nmx_iguana.png','PNG')  #바꾼배열을 다시 이미지화 시켜 png파일로 저장한다.

    return n_m_s

def double_thresholding(img):
    
    #threshold할 두 기준점을 정의한다.
    diff = img.max() - img.min()
    T_high = img.min() + diff*0.15
    T_low = img.min() + diff*0.03 

    #Thigh보다 높은 값을 가지는 pixel은 255로 T_hight, T_low사이 값을 가지는 pixel은 80으로 나머지는 0으로 만든다.
    d_t = np.copy(img)
    d_t = np.where(d_t>T_high,255,d_t)
    d_t = np.where((T_low <= d_t) & (d_t < T_high),80,d_t)
    d_t = np.where(d_t <T_low,0,d_t)

    d_t_img = Image.fromarray(d_t.astype('uint8'))
    d_t_img.save('d_t_iguana.png','PNG')  #바꾼배열을 다시 이미지화 시켜 png파일로 저장한다.

    return d_t

def hysteresis(img):

    #pixel주변 pixel을 탐색할때 사용될 index들
    dx = [0,1,1,1,0,-1,-1,-1] 
    dy = [1,1,0,-1,-1,-1,0,1]

    #img배열의 copy를 만들고 pixel값이 255인 좌표를 stack에 넣는다.
    hys = np.copy(img)
    stack = []
    h,w = img.shape
    for i in range(1,h-1):
        for j in range(1,w-1):
            if img[i][j] == 255:
                stack.append((i, j))

    #stack이 빌때까지 stack에서 pop한 좌표주변을 탐색하여 그 값이 80인경우 255로 만들고 다시 그 좌표를 stack에 넣는다.
    while stack:
        i,j = stack.pop()
        for a in range(8):
            if hys[i+dx[a]][j+dy[a]] == 80:
                hys[i+dx[a]][j+dy[a]] = 255
                stack.append((i+dx[a],j+dy[a]))
        
    #끝내 255가 되지 못하고 80값을 가지는 pixel은 strong edge와 연결이 되어있지 않다는 뜻이므로 0으로 만들어준다. 
    hys = np.where(hys == 80 , 0 , hys)

    hys_img = Image.fromarray(hys.astype('uint8'))
    hys_img.save('hys_iguana.png','PNG')  #바꾼배열을 다시 이미지화 시켜 png파일로 저장한다.  
    return hys




def main():
    im = Image.open('iguana.bmp') #이구아나 사진을 불러온다.
    im = im.convert('L')   #사진을 흑백으로 전환한다.
    im.save("gray_version_iguana.png", 'PNG')
    im_array = np.asarray(im)
    gauss_array = gaussconvolve2d(im_array,1.6)
    int_gauss_array = gauss_array.astype('uint8')  # 추출한 배열을 sigma 1.6인 가우스필터를 적용하고 uint형 배열로 바꾼다.
    im4 = Image.fromarray(int_gauss_array)
    im4.save('low_passed_iguana.png','PNG')  #바꾼배열을 다시 이미지화 시켜 png파일로 저장한다.

    G, theta = sobel_filters(gauss_array)
    nms = non_max_suppression(G,theta)
    d_t = double_thresholding(nms)
    hysteresis(d_t)

main()
유튜브 영상 : 10분안에 배우는 머신러닝 GAN(생성적 적대적 신경망) 알고리즘~~~ : https://www.youtube.com/watch?v=N9ewzLUZhL8
+ https://www.youtube.com/user/hunkims(김성훈(홍콩과기대) 딥러닝) : 책 : 모두의 딥러닝 v2 / 밑바닥부터 시작하는 딥러닝 / 신경망 첫걸음 / 딥러닝 첫걸음 / 미술관에 GAN 딥러닝 실천 프로젝트

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

GAN 관련 피피티 내용

Generative model
Generative Adverserial Network (생성적 적대 신경망, 적대적 생성 신경망)

생성자는 판별자가 구별할 수 없을 정도의 정교한 가짜를 만드는 것이 목적 / 가우시안 필터
생정자 : 가짜 제조 공장
가상의 이미지를 만들어 내는 공장임  / 성능 자체를 높이기 위해서 다양한 로스(loss)를 사용함 → 그 값을 통해 가중치 업데이트
→ 작동방식 : 처음엔 랜덤한 픽셀 값으로  채워진 가짜이미지로    시작 → 판별자의 판별 결과에 따라서 지속적으로 업데이트 → 점차 원하는 이미지를 만들어감

DICRIMIATOR(판별자, 진위판별장치) : 생성자에서 넘어온 이미지가 가짜인지 진짜인지 판별해주는 장지(진짜 = 1, 가자 = 1)

Generative adversaria networks(gans) similar with AEs
- learn dense representations
- can be used as nerative models

GAN applicatios:
- super resulution, colorization, image editing(deep fake +)
- predicting the next frames in a video(낮은 프레임(24 frame)을 게임용 화면용으로 만들기 위해서 높은 프레임(64, 144frame)
- augmenting a dataset(데이터 증강)
- generating other types of data(다른 타입의 데이터를 만들어낼 수 있음)
- identifying the weaknesses in otehr models and strenthening them

단점 : mode collaps : 분포 중 끝에 있는 데이터를 학습하기 어려운 것이 단점(적은 다양성) / 즉각적으로 데이터가 안나온다, 제로섬 게임

하나의 해결책(완벽하지 않음) : mini-batch discrimination(다양성을 얻기 위해)


DCGAN, StyleGAN(옷, 머리스타일 등 의 디자인을 변형시킬 수 있음)
DCGAN(Deep Convolutional GANs) 이런게 있다~ : convolutional layer를 많이 쌓아서 dcgan

PGGAN(Progressive Growing of GAN) : 점차 발전해가면서 GAN... / AutoEncoder랑 유사한 구조

StyleGAN

MuseGAN : 곡의 특징을 살려서 음악을 만들어냄(쇼팽, 베토벤) / 이미지도 가능(고흐 등)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

celebA
https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
드라이브안의 내용들 다 다운

슬랙의 깃허브에 다운로드
바탕화면에 pytorch / data(폴더생성) / celela(폴더생성) / img_align_ celeba(폴더이동)

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 튜토리얼 코딩 실습


sigmoid, tanh는 기울기 소실이라는 문제가 발생함 → ReLU, Leaky ReLU를 흔히 씀					
GAN의 평가지표 2가지(IS(Inception Score / FID(Frechet Inception Distance)

Image Quality Assessment(GAN을 통해 나온 결과이미지를 평가 지표)
SSIM(Structure Smiliarity)(외우기)
PSNR(Peak signal to noise ratio)(외우기//)
GMSD(Gradient Magnitude Similiarity Deviation)
LPIPS(Learned Perceptual Image Patch Similiarity)
mos (Mean opinion score) : 사람이 직접 평가를 내려서 의견이 들어간 점수


ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
SRGAN

cd C:\Users\23\Desktop\pytorch\SRGAN

https://github.com/leftthomas/SRGAN

SRGAN\data
Data폴더로 옮기기 → DIV2K_train_HR → 000001.jp / 000002.jpg


conda install tqdm
→ progress bar를 보여주는 라이브러리

python train.py

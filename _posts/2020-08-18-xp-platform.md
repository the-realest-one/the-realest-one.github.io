---
layout: post
title:  "Presto - Distributed SQL Query Engine on Multiple Sources"
date:   2020-08-18T11:11:11
author: the-realest-one
categories: DataPlatform
tags:	Uber XP ExperimentalPlatform DataScience
cover:  "/assets/North.jpg"
use_math: true
---

# Experimental system in Uber

Building an Intelligent Experimentation Platform with Uber Engineering

https://eng.uber.com/experimentation-platform/

## XP 시스템의 필요 이유

어떤 기능을 낼 때, 이 기능이 괜찮은지 아닌지 알아야해. 기능 전 세계인한테 딱 냈는데 쓰레기면 망하는 거지.
XP 역할: 새로운 기능을 잘 rollout 하고, 이 기능이 괜찮은지 분석하게 하는 것.

이 두 역할을 위해 우버는 어떻게 했다?
1. Staged rollout (안 좋으면 빼고, 좋으면 계속 ㄱ)
2. Intelligent analysis (안 좋은지 좋은지 알아야지)
로 XP 시스템을 구성했다!

## Staged rollout

새로운 피처를 단계적으로 출시하는 것.
A/B test 랑은 다른 거임! A/B 테스트는 새로운 피쳐를 만들까? 말까? 인거고, XP 는 만든 피쳐를 출시하는 거임.
출시가 잘 되는지 보기 위해 core app health 와 business metric 을 계속 모니터링함.

!질문! Staged rollout 의 골이 “Whether the feature is causing a regression” 이라는데 무슨 뜻일까

### Architecting the New Feature Rollout System

우버 XP 컴포넌트: a staged rollout configuration process +  a new monitoring algorithm.

귀무가설 H0: 새 피쳐는 key metric 에 부정적인 영향이 없다.
Key metric 에 영향을 유의미하게 미치는 지 보기 위해, t-test, a sequential likelihood ratio test (SLRT), and a delete-a-group jackknife variance estimation 라는 걸 쓴다고 한다.
이것들이 뭔지는 나중에 찾아보자.

처음에는 t-test 썼는데 별로 안좋았다는 듯? T-test 는 continuous monitoring 을 위한 게 아니거든.
그리고 나서, regression 을 더 정확하게 detect 하기 위해 independence assumption 과 함께 SLRT 을 써봤다.

SLRT with independence assumption 를 써도 30 % 의 false positive 를 줌.
왜 이렇게 만족스럽지 않은 결과가 나왔는가?
우버의 metric 들은 session 레벨에서 계산되는데, 만약 동일한 유저라면 두 개의 세션의 행동이 상관관계가 매우 큼. 즉 독립이 아닌 거지.
마지막으로 쓴 게 SLRT using the delete-a-group jackknife variance. 이거는 5 % 의 false positive 를 줬대.

### Achievement and Outcomes of Staged rollout

써보니까 괜찮아서, 우버의 다른 프로덕트들에도 적용됨.
그리고 rollout 초기 단계에서 regressions caused by features 을 잡아내면서, 유저에게 주는 안 좋은 임팩트를 줄였다.

성공 예시: user name 이 아니라 핸드폰 번호로 로그인하는 피쳐를 출시했음.
그런데 갑자기 특정 지역에서 trip rate 가 급감하는 걸 발견함.
왜인고 봤더니, 왜인지 treatment group 의 유저들이 핸드폰 번호로 로그인을 못하고 있던 거임!
이걸 XP 프로그램덕분에 모든 유저에게 outage 가 발생하기 전에 발견했다!

## Intelligent Analysis Tool

먼저. Intelligent Analysis Tool in Uber 란?
rollout 이후, real-time experiment result 를 주는 툴.

왜 필요했나? 계속 새로운 도시로 확장하고, 또 이미 있는 도시에서도 사업을 확장하고 있음 -> real-time experiment results 를 줄 수 있어야했음.
예시: 운영 팀은. 각 도시마다 살짝씩 문자 메세지를 뭐 fine tuning 했다? 잘 모르겠음

또 문제. 운영 팀과 XP 팀의 시간대가 달라서, 운영팀과 XP팀이 협력해서 rollout 후 분석 보고서를 run 하는 데 오래 걸렸다.
(아마 뭐 분석 보고서 만들기? 인듯?).
우리의 새로운 프레임워크로, 실행시간도 엄청 줄이고, 운영팀이 어느 시간대에 살든 실험 결과를 볼 수 있게 만들었다!


### Defining the XP Architecture: Intelligent Analysis Tool

처음에는 분석 속도를 빠르게 하려고, 비즈니스 메트릭의 통계값들을 Hive 에서 pre-compute 했음.
즉, ETL 을 한 번 거친 것을 Analysis Tool 에 준 것이다.
그런데, 이건 end-user (아마 운영팀? 혹은 분석가)가  메트릭 정의를 customize 할 수가 없음. 즉, 유연성이 없는 거지.

새로운 분석 툴은 데이터를 pre-compute 하지 않아. -> 스토리지 지출 줄고, 분석 시간 단축. (이거 용으로 ETL 을 따로 안한다는 뜻인듯?)
(raw event 들을 모두 넣으면 너무 부하가 많이 걸릴 것 같은데, 어떻게 해결한 걸까.)
그리고 지금은, 유저가 SQL 파일로 메트릭을 바로 생성해.
유저가 SQL 을 webUI 에 입력하고, 그거에 따라 sql 로 메트릭을 만드는 듯.

그리고 서비스 엔진으로 스칼라를 써서 treatment group 과 control group 의 평균이 다른지 아닌지 p-value 계산.
이거로 expriment 가 타겟 사이즈에 도달했는지 본다. // TODO 이 부분은 설명이 자세하지가 않아서 잘 모르겠어.

## Science After XP

XP 다음으로 post-experiment analysis 도 진행한다.
목표는 treatment group 이 control group 에 비해 많은 상승이 있었는지 보는 것.
이를 위해 메트릭들을 proportion metrics, continuous metrics, and ratio metrics 이렇게 3 개로 나눴다.
뭐 어떤 어떤 메트릭을 어케 나눴는지는 쓰지 않는다. 링크를 따라가보자.

## Conclusion

이렇게 uber XP 팀은, staged rollout 과 post-experiment analysis 로 앱에 stability 를 주고 있다!

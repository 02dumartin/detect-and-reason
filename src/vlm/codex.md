* 코드 작성 시 명시
md 파일의 지시를 먼저 따름

- 코드 수정/추가/제거 시
    어떤 폴더의
    어떤 파일의
    어떤 함수의
    어떤 방식으로 수정/추가/제거 해야 하는지를 정확하게 지시



- 코드 수정/추거/제거에 관한 논의를 한다면

1) 반드시 계획을 먼저 세움 

2) 계획은 다음을 명시
    어떤 폴더의
    어떤 파일의
    어떤 함수의
    어떤 방식으로 수정/추가/제거 해야 하는지 

3) 코드 추가/수정/제거 지시가 없을 경우에 임의로 추가하지 않음




===



* 현재 변경/추가 해야 하는 것
- adjust module

- qwen sft 결과 확인 및 이후 수정
    decoder only SFT adapter 학습 잘 됨
    merge
    or qwen_hf 에서 adapter를 직접 로드
    두 개가 어떤 차이인건지 확인하기

- qwen sft 할 때, 프롬프트 + example로 학습할 수 있는지 알아보기 

- RF 1cls result + VLM: custom_tomato, Rbo2Pheno 1cls

- zeroshot sh 실행
    g dino, owl vit, yolo world zero shot: custom_tomato, Rbo2Pheno 2cls
    
    예전 config
    GDINO = 0.35 / 0.25 / IoU 0.5
    OWL-ViT = 0.10 / IoU 0.5
    YOLO-World = 0.05 / IoU 0.5 // confidence score말고, 또 다른 파라미터는 없는지 확인



* 필요한 실험 다시 정리
실험(train-test data 기준 / prediction path / vlm path)
use reasoning, example 사용, SFT 가중치 사용

# table1 추가 실험
tomatod-tomatod rf-detr 1cls 
result/detection_reasoning/tomatod/rf_detr_1cls_prediction/test/predictions_coco.json
result/detection_reasoning/tomatod/rf_detr_1cls_qwen

little-little rf-detr 1cls
result/detection_reasoning/little/rf_detr_1cls_prediction/test/predictions_coco.json
result/detection_reasoning/little/rf_detr_1cls_qwen

merge-tomatod rf-detr 1cls 
result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_prediction/test/predictions_coco.json
result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_qwen

merge-little rf-detr 1cls
result/detection_reasoning/merge/little/rf_detr_1cls_little_prediction/test/predictions_coco.json
result/detection_reasoning/merge/little/rf_detr_1cls_little_qwen


# table4 추가 실험
rob2pheno-rob2pheno rf-detr 1cls
result/detection_reasoning/rob2pheno/rf_detr_1cls_perdiction/test/predictions_coco.json
result/detection_reasoning/rob2pheno/rf_detr_1cls_qwen


custom tomato-custom tomato rf-detr 1cls
result/detection_reasoning/custom_tomato/rf_detr_1cls_prediction/test/predictions_coco.json
result/detection_reasoning/custom_tomato/rf_detr_1cls_qwen




* 보류한 것
- yolo world class 설정에서, {red tomato, green tomato, ""} 이렇게 쓰는 방식 

- sh 파일 만드는 문법
이걸 완전히 의존하니까 하고 싶은 거나, 오류가 생겼을 때 뭐 때문에 오류인지를 몰겠음

- Examlpe RAG 적용
example_picker.py와 연관
현재의 방식에 추가하고 싶은 것 / 현재를 대체하는 것이 아님

아이디어안) 현재 target crop과 시각적으로 비슷한 exemplar를 그때그때 찾아서 넣음
Offline exemplar bank 구축train split의 GT tomato crop들을 exemplar pool로 만듦

각 exemplar에 다음을 저장crop image
label (fully-ripe, semi-ripe, unripe)
source image / stem / row index
embedding vector
선택적으로 brightness, saturation, blur, bbox size 같은 보조 메타데이터


Query crop 임베딩 생성현재 detector crop을 같은 encoder로 embedding
예: CLIP / SigLIP / DINOv2 계열처럼 visual similarity에 강한 encoder

Retrievalquery와 exemplar bank 사이 cosine similarity로 nearest neighbors 검색
여기서 가져온 후보가 “example RAG 결과”

Prompt용 exemplar selectionretrieval 결과를 그대로 다 넣는 게 아니라, 최종 prompt에 넣을 exemplar를 다시 고름
이유:top-k만 그대로 넣으면 한 클래스만 몰릴 수 있음
조명/배경만 비슷하고 ripeness 기준은 덜 좋은 exemplar가 섞일 수 있음


Prompt assembly최종 선정 exemplar를 지금 구조의 examples 자리에 넣음
즉 [prompting.py (line 26)](/home/hyeonjin/detect-and-reason/src/vlm/prompting.py:26)의 use_examples 흐름은 유지하고, “어떤 examples를 넣을지”만 smarter하게 바꾸는 개념

추천  retrieval 방식
가장 현실적인 건 “2단계"
1단계: similarity 기반으로 후보 top-M retrieval
2단계: 그 후보들 중에서 prompt용 top-K를 재선정
I have used pretrained bert-base-uncased available through huggingface and used linear probing for predicting NER.

I also attempted the optional task of implementing CRF+Viterbi alogirhtm.

My main model trains for 10 epochs and the CRF model trains for 3 epochs. If GPUs are used then it should be done within 10 min. 

I have also presented the macro f1 and overall classification report computation, based on the ed post #71


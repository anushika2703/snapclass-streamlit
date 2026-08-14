from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import io
import librosa #librosa 2 cheezein return krta hai audio or sample rate
import streamlit as st


@st.cache_resource 
def load_voice_encoder():
    return VoiceEncoder()

def get_voice_embedding(audio_bytes):
    try:
        encoder=load_voice_encoder()
        
        #jitna zyda sample audio hot ahai utna jyda defined audio hota hai but ofc utna jyda time bhi lgta
        audio,sr = librosa.load(io.BytesIO(audio_bytes),sr=16000)
        wav=preprocess_wav(audio)
        #embed_utterance krke function hai voh voice ki embedding de deta hai
        embedding=encoder.embed_utterance(wav)
        return embedding.tolist()  #voice mein 256 2D vector hota h
    except Exception as e:
        st.error('Voice recog error')
        return None
    
# abb hum next speaker ko identify krne ke liye function bnayenge
#ismein hum jo voice embedding ayegi usko sbki students ki stored voice se compare kr lenge; ek threshold define kr lenge
#hum dot product nikalte h or usse similarity search krte hai
def identify_speaker(new_embedding, candidates_dict, threshold=0.65):
    #agr student ki awaz hai hi nhi dictionary mein
    if new_embedding is None or not candidates_dict:
        return None, 0
    best_sid = None
    best_score=-1.0
    #why dot product?
    for sid,stored_embedding in candidates_dict.items():
        if stored_embedding:
            similarity=np.dot(new_embedding,stored_embedding)
            if similarity>=best_score:
                best_score=similarity
                best_sid=sid
    if best_score>=threshold:
        return best_sid, best_score
    return None, best_score

#abb yeh jo humne uor kiya hai usko abb hum bulk mein krenge
#teacher apna mic khol deta hai or students baari baari aakr present bolte hai
#toh yahi audio bulk mein aa jata hai toh abb hum usko tukdo mein divide krenge

def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.65):
    try:
        encoder=load_voice_encoder()
        audio,sr=librosa.load(io.BytesIO(audio_bytes),sr=16000)
        #top_db -> sensitivity bta rhi hai voice ki; agr high krenge toh sorf jbb students chilla rhe honge tbb hi record hoga agr small kr denge toh abhut low voice catch krega jismein
        #errors ki possibility jyda hai; isliye hum ek middle range rkh rhe hai
        segments=librosa.effects.split(audio,top_db=30)

        identified_results={}

        for start, end in segments:
            #jo bhi voice bahut choti si ati ahi ya dheere ati ahi usko hum remove kr denge
            if(end-start) < sr*0.5:
                continue
            #abb hum segment se start or end of audio  nikal lenge for a particular student
            segment_audio=audio[start:end]   
            wav=preprocess_wav(segment_audio)
            embedding=encoder.embed_utterance(wav)

            sid,score=identify_speaker(embedding,candidates_dict, threshold)

            if sid:
                if sid not in identified_results or score > identified_results[sid]:
                    identified_results[sid]=score
        return identified_results
    except Exception as e:
        st.error('Bulk process error')
        return {}


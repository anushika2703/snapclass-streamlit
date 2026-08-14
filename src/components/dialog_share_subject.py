import streamlit as st
import segno #QR generate krne ke liye chahiye hoga
import io

@st.dialog("Share Class Link")
def share_subject_dialog(subject_name, subject_code):
   app_domain = "http://localhost:8501"
   join_url = f"{app_domain.rstrip('/')}/?join-code={subject_code}"

   st.header("Scan to Join")

   #QR bnane ke liye
   qr = segno.make(join_url)
   #output le lete hai
   out=io.BytesIO()
   #qr ko save krne ke liye
   qr.save(out, kind='png', scale=10, border=1)

   col1,col2=st.columns(2)
   with col1:
      st.markdown('### Copy Link')
      st.code(join_url, language="text")
      st.code(subject_code,language="text")
      st.info('Copy this link to share on Whatsapp or Email')
   with col2:
      st.markdown('### Scan to Join')
      st.image(out.getvalue(), caption='QRCODE for class joining')
      
      


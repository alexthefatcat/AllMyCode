# -*- coding: utf-8 -*-
"""Created on Tue Dec 17 10:34:34 2019@author: Alexm"""

def smtp_gmail(email_to="dogman@hotmail.com",subject_line= "This is my subject line!",body_text= "This is my message that can also have a dog"):
    import smtplib
    username    = "circalerrorreport@gmail.com"
    password    = "GMcircal<doorcode>"
    smtp_server = ("smtp.gmail.com", 587)
    email_from  = "circalerrorreport@gmail.com"

    email_body = "\r\n".join([f"From: {email_from}",f"To: {email_to}",f"Subject: {subject_line}","",body_text])

    server = smtplib.SMTP(*smtp_server)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(username, password) 
    server.sendmail(email_from, email_to, email_body)
    server.quit()
    
    
#dob:ddmmyyyy   (y1y2)(y1y2y3)(y4+3)
if False:
    # -*- coding: utf-8 -*-
  """Created on Tue Dec 10 10:32:50 2019  @author: Alexm  """

   # Import smtplib for the actual sending function
   import smtplib
   from email.message import EmailMessage

   msg            = EmailMessage()
   msg['Subject'] = 'Example email'
   msg['From'   ] = "test@hotmail.co.uk"
   msg['To'     ] = ', '.join(["dogman@hotmail.com"])
   msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'

   with smtplib.SMTP('localhost') as s:
       s.send_message(msg)














    
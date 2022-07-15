from tkinter import*
from PIL import ImageTk #pip install pilo 
from tkinter import ttk,messagebox
import pymysql #pip install pymysql

class Login:
    def __init__(self,root):
        self.root=root
        self.root.title("Login Page")
        self.root.geometry("1100x600+100+50")
        self.root.resizable(False,False)
        
        
        self.bg=ImageTk.PhotoImage(file="bg.jpg")
        self.bg_image=Label(self.root,image=self.bg).place(x=0,y=0,relwidth=1,relheight=1)
        
        #===Login Frame====
        
        Frame_login=Frame(self.root,bg="white")
        Frame_login.place(x=150,y=150,height=340,width=500)
        
        
        title=Label(Frame_login,text="login Here",font=("Impact",35,"bold"),fg="#d77337",bg="white").place(x=90,y=30)
        desc=Label(Frame_login,text="Sarcasm Detection Login Area",font=("Goudy old Style",15,"bold"),fg="#d25d17",bg="white").place(x=90,y=100)
        
        lbl_Email=Label(Frame_login,text="Username",font=("Goudy old Style",15,"bold"),fg="grey",bg="white").place(x=90,y=140)
        self.txt_Email=Entry(Frame_login,font=("time new roman",15),bg="lightgray")
        self.txt_Email.place(x=90,y=170,width=350,height=35)
        
        lbl_pass=Label(Frame_login,text="Password",font=("Goudy old Style",15,"bold"),fg="grey",bg="white").place(x=90,y=210)
        self.txt_pass=Entry(Frame_login,font=("time new roman",15),bg="lightgray")
        self.txt_pass.place(x=90,y=240,width=350,height=35)
        
        
        forget_btn=Button(Frame_login,text="Register Now?",command=self.register_window,bg="white",fg="#d77337",bd=0,font=("time new Roman",12)).place(x=90,y=280)
        login_btn=Button(self.root,text="Login",command=self.login,bg="#d77337",fg="white",font=("time new Roman",20)).place(x=300,y=470,width=180,height=40)
        
    
    def register_window(self):
        self.root.destroy()
        import register
        
    def login(self):
        if self.txt_Email.get()==""or self.txt_pass.get()=="":
           messagebox.showerror("Error","all Fields are Required",parent=self.root)
        else:
            
            try:
                con=pymysql.connect(host="localhost",user="root",password="",database="mypro")
                cur=con.cursor()
                cur.execute("select * from user where Email=%s and password=%s",(self.txt_Email.get(),self.txt_pass.get()))
                row=cur.fetchone()
                #print(row)
                
                if row==None:
                    messagebox.showerror("Error","Invalid UserName And PAssword",parent=self.root)
                    
                else :
                    messagebox.showinfo("success","Login success",parent=self.root)
            except Exception as es:
                
                messagebox.showerror("Error",f"Error Duw to:{str(es)}",parent=self.root)
        
root=Tk()      
obj=Login(root)
root.mainloop()
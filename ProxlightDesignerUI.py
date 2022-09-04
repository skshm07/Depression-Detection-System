
from tkinter import *
from tkinter import filedialog, messagebox
from tkcalendar import DateEntry as tkDateEntry
import tkmagicgrid as tkMG
import tkscrolledframe as tkSF
from PIL import Image, ImageTk
import io
import csv
import json
import datetime

from Core import Helper as Hlpr
from Core import Predict

##################################################################
## Init
wnd_width, wnd_height = 1000, 600
window = Tk()
window.title("Twitter Depression Detector")
#window.iconbitmap("Icon.ico")
window.geometry(f"{wnd_width}x{wnd_height}")
window.configure(bg = "#FFFFFF")
canvas = Canvas(
    window,
    bg = "#FFFFFF",
    width = wnd_width,
    height = wnd_height,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

textbox_background_img = ImageTk.PhotoImage(Image.open("img_textBox1.png"))
textbox_background_img_white = ImageTk.PhotoImage(Image.open("img_textBox0.png"))
run_img = ImageTk.PhotoImage(Image.open("run_button2.jpg").resize((178,63), Image.Resampling.LANCZOS))
textbox_background_img_create = ImageTk.PhotoImage(Image.open("img0.png").resize((140,55), Image.Resampling.LANCZOS))

background_img = ImageTk.PhotoImage(Image.open("background.jpg").resize((wnd_width,wnd_height), Image.Resampling.LANCZOS))
twitter_img = ImageTk.PhotoImage(Image.open("logo2.png").resize((423,298), Image.Resampling.LANCZOS))

background_color = "#7e7f79"
background_color2 = "#8e8f89"

background = canvas.create_image(
    wnd_width/2, wnd_height/2,
    image = background_img)
twitter_logo = canvas.create_image(
    200, 150,
    image = twitter_img)

my_font = ("Roboto-Light", int(14.0))

######################################D
username_found_status = None
username_entry_sv = StringVar()
x, y = 660, 60
canvas.create_text(
    x, y,
    text = "twitter Username",
    fill = "cyan",
    font = my_font)
username_found_status = canvas.create_text(
    x + 90, y + 85,
    text = "??",
    fill = "#ffffff",
    font = ("consolas", int(10.0)))
canvas.create_image(
    x + 83.5, y + 43.5,
    image = textbox_background_img)
Entry(
    bd = 0,
    bg = "#cfdcd9",
    highlightthickness = 0,
    textvariable = username_entry_sv).place(
        x = x - 47, y = y + 19,
        width = 268.0,
        height = 45)
######################################D
g_StartDate, g_EndDate = StringVar(), StringVar()
def my_DateEntry(x, y, the_text, textvariable):
    canvas.create_text(
        x - (len(the_text)*7)-34, y,
        text = the_text,
        fill = "cyan",
        font = my_font)
    dete_entry_bg = canvas.create_image(
        x + 36.5, y +  5,
        image = textbox_background_img_create)
    dete_entry = tkDateEntry(
        text = the_text, 
        date_pattern = "yyyy-MM-dd", 
        textvariable = textvariable)
    dete_entry.place(
        x = x - 30, y = y - 19.5,
        width = 133,
        height = 49)
    return dete_entry
x, y = x - 40, y + 125
start_date = my_DateEntry(x, y, "from", g_StartDate)
end_date = my_DateEntry(x + 180, y, "to", g_EndDate)
start_date.set_date(end_date.get_date() - datetime.timedelta(days=1))
######################################D
x, y = x + 100, y + 50
run_analysis = Button(
    image = run_img,
    borderwidth = 0,
    highlightthickness = 0,
    relief = "flat")

run_analysis.place(
    x = x, y = y,
    width = 178,
    height = 63)
######################################D
count_ratio_handle = None
deppressive_tweet_count_handle = None
non_deppressive_tweet_count_handle = None

x, y = x - 260, y + 15
canvas.create_text(
    x, y,
    text = "/Stats:        T \\",
    fill = background_color,
    font = my_font) 
count_ratio_handle = canvas.create_text(
    x + 15, y,
    text = "0",
    fill = "pink",
    font = my_font) 
x, y = x - 80, y + 25 
canvas.create_text(
    x, y,
    text = "Depressive:",
    fill = background_color,
    font = my_font) 
deppressive_tweet_count_handle = canvas.create_text(
    x + 70, y,
    text = "0",
    fill = "red",
    font = my_font)
x, y = x + 180, y  
canvas.create_text(
    x, y,
    text = "Non-Depressive:",
    fill = background_color,
    font = my_font)
non_deppressive_tweet_count_handle = canvas.create_text(
    x + 90, y,
    text = "0",
    fill = "green",
    font = my_font)
######################################D
# Create a MagicGrid widget
x, y = 50, 320
sf = tkSF.ScrolledFrame(canvas)
sf.place(x = x, y = y, width = 900, height = 250)
sf.bind_arrow_keys(canvas)
sf.bind_scroll_wheel(canvas)
tweets_viewer = tkMG.MagicGrid(sf.display_widget(Frame))
tweets_viewer._bg_header   = "black"
tweets_viewer._bg_color  = background_color
tweets_viewer._bg_shade  = background_color2
def updateDisplayTable (csv_path):
    global tweets_viewer

    for child in tweets_viewer.winfo_children():
        child.destroy()

    with io.open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        parse_iter = iter(reader)

        tweets_viewer.add_header(*next(parse_iter))

        for row in parse_iter:
        	tweets_viewer.add_row(*row)

updateDisplayTable ("tweets.csv")
tweets_viewer.grid(row=0,column=0)

#######################################
g_TargetUserID, g_OkUsername = 0, False

# methods
def search_username(sv):
    global g_TargetUserID, g_OkUsername
    global canvas, username_found_status

    username = sv.get()
    g_OkUsername = False

    if username == "":
        canvas.itemconfigure(username_found_status, text="no username")
        canvas.itemconfigure(username_found_status, fill="white")
        return

    url = "https://api.twitter.com/2/users/by?usernames={}".format(username)
    try:
        json_response = Hlpr.connectToTwitterEndpoint(url)
        
    except Exception as err:
        err_txt = str(err)
        if len(err_txt) > 50:
            err_txt = err_txt[:47] + '...'
        canvas.itemconfigure(username_found_status, text=err_txt)
        canvas.itemconfigure(username_found_status, fill="red")
        print(err)
        print('----------------')
        return

    if ("errors" in json_response):
        out = json_response['errors'][0]['detail']
    elif ('data' in json_response):
        out = json_response['data'][0]['id']
        g_TargetUserID = int(out, 10)
        g_OkUsername = True
    else:
        out = "unknown error"

    if g_OkUsername:
        canvas.itemconfigure(username_found_status, text = f"UserID = {out}")
        canvas.itemconfigure(username_found_status, fill = "green")
    else:
        canvas.itemconfigure(username_found_status, text = out)
        canvas.itemconfigure(username_found_status, fill = "red")
    
username_entry_sv.trace("w", lambda name, index, mode, sv=username_entry_sv: search_username(sv))

def UpdateStartDateMax(e):
    global canvas, start_date, end_date
    max_date = end_date.get_date() - datetime.timedelta(days=1)
    if start_date.get_date() > max_date:
        start_date.set_date(max_date)

def UpdateEndDateMin(e):
    global canvas, start_date, end_date
    min_date = start_date.get_date() + datetime.timedelta(days=1)
    if end_date.get_date() < min_date:
        end_date.set_date (min_date)

start_date.bind("<<DateEntrySelected>>", UpdateStartDateMax)
end_date.bind("<<DateEntrySelected>>", UpdateEndDateMin)

def getTweetsAndReplies(e):
    global g_TargetUserID, g_OkUsername, g_StartDate, g_EndDate
    global canvas, deppressive_tweet_count_handle, non_deppressive_tweet_count_handle

    filter_fields = "start_time={}T00%3A00%3A00%2B05%3A30&end_time={}T00%3A00%3A00%2B05%3A30&max_results=100".format(g_StartDate.get(), g_EndDate.get())
    if not g_OkUsername:
        return

    
    url = "https://api.twitter.com/2/users/{}/tweets?{}".format(g_TargetUserID, filter_fields)
    json_response = Hlpr.connectToTwitterEndpoint(url)

    result_count = 0
    if ("errors" in json_response):
        for errs in json_response['errors']:
            for _err_ in errs:
                out += _err_['detail'] + "; "
    elif ('data' in json_response):
        out = json_response['data']
        out_file = str(Hlpr.TweetsJSONResponseToCSVFile(out))

        result_count = json_response['meta']['result_count']
        out = json.dumps(out, indent=4, sort_keys=True)
        
        out_csv_path = Predict.evaluate(out_file)
        with io.open(out_csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            parse_iter = iter(reader)
            header = next(parse_iter)
            if "target" in header:
                index = header.index("target")
                depression_count = 0
                total_count = 0
                for row in parse_iter:
                    depression_count += int(row[index], 10)
                    total_count += 1
                canvas.itemconfigure(deppressive_tweet_count_handle, text=f"{depression_count}")
                canvas.itemconfigure(non_deppressive_tweet_count_handle, text=f"{total_count - depression_count}")
                canvas.itemconfigure(count_ratio_handle, text=f"{total_count}")
        updateDisplayTable(out_csv_path)

    else:
        out = "unknown error"

    print("################################################")
    print(out, "\nresult_count = {}\n".format(result_count), filter_fields)

canvas.itemconfigure(deppressive_tweet_count_handle, text="55")
canvas.itemconfigure(non_deppressive_tweet_count_handle, text="55")
run_analysis.bind("<Button-1>", getTweetsAndReplies)
#canvas.itemconfigure(username_found_status, text="")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

embedding_dim = 100
n_hidden = 64
n_out = 2
class ConcatPoolingGRUAdaptive(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out, pretrained_vec, dropout, bidirectional=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bidirectional = bidirectional
        
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec)
        self.emb.weight.requires_grad = False
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(self.n_hidden*2*2, self.n_out)
        else:
            self.fc = nn.Linear(self.n_hidden*2, self.n_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq, lengths):
        bs = seq.size(1)
        self.h = self.init_hidden(bs)
        seq = seq.transpose(0,1)
        embs = self.emb(seq)
        embs = embs.transpose(0,1)
        embs = pack_padded_sequence(embs, lengths)
        gru_out, self.h = self.gru(embs, self.h)
        gru_out, lengths = pad_packed_sequence(gru_out)        
        
        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0),1).view(bs,-1) 
        
        cat = self.dropout(torch.cat([avg_pool,max_pool],dim=1))
        
        outp = self.fc(cat)
        return F.log_softmax(outp)
    
    def init_hidden(self, batch_size): 
        if self.bidirectional:
            return torch.zeros((2,batch_size,self.n_hidden)).to(Predict.device)
        else:
            return torch.zeros((1,batch_size,self.n_hidden)).cuda().to(Predict.device)

Predict.init()
window.resizable(False, False)
window.mainloop()

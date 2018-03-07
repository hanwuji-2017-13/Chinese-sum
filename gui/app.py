import wx
from textsum import Frame
from textsum import SumingDialog
from textsum import AboutDialog
import os,sys    
import _thread
import threading
import wx.html2
from abc import ABCMeta, abstractmethod  

class CountEvent(wx.PyCommandEvent):
    def __init__(self,etype, value=None):
        super().__init__(etype)
        self._value = value

    def GetValue(self):
        return self._value

class async_run:
    def __init__(self,func,OnFinish):
        self.id=wx.NewEventType()
        self.func=func
        self.OnFinish_imp=OnFinish
        app.Bind(wx.PyEventBinder(self.id),self.OnFinish)

    def run(self,arg):
        _thread.start_new_thread(self.OnRun,(arg,))

    def OnRun(self,arg):
        value=self.func(*arg)
        wx.PostEvent(app,CountEvent(self.id,value))

    def OnFinish(self,evt):
        if(evt.GetValue()!=None):
            self.OnFinish_imp(evt.GetValue())
        else:
            self.OnFinish_imp()
        
class sumFrame ( Frame ):
	
    def __init__( self):
        super().__init__(None)
        self.m_bpButton1.Enable(False)
        self.m_statusBar1.SetStatusText("loading data")
        self.async_GenTextSum=async_run(textSumApi.OnGenTextSum,self.OnGenTextSumFin)
        self.async_SelTextSum=async_run(textSumApi.OnSelTextSum,self.OnSelTextSumFin)

    def OnTextSum(self,event):
        self.loaing(True)
        self.async_SelTextSum.run((self.m_textCtrl1.GetValue(),));
        self.sd=SumingDialog(self)
        self.sd.m_staticText5.SetLabel("正在生成摘要（摘取式）")
        self.sd.Show(True)
        self.sd.GetSizer().Fit(self.sd)
        self.sd.ShowModal()

    def OnSelTextSumFin(self,value):
        self.async_GenTextSum.run((self.m_textCtrl1.GetValue(),))
        self.m_textCtrl2.SetValue(value[0])
        self.m_staticText1.SetLabel("rouge:"+str(value[1]))
        self.sd.m_staticText5.SetLabel("正在生成摘要（生成式）")
        webv=wx.html2.WebView.New(self.sd)
        self.sd.GetSizer().Add(webv,1, wx.ALL|wx.EXPAND, 5)
        webv.LoadURL(textSumApi.get_html_url())
        self.sd.SetSize(500,500)
        self.sd.Layout()
        self.Refresh()
        
    def OnGenTextSumFin(self,value):
        self.sd.Close()
        self.m_textCtrl3.SetValue(value[0])
        self.m_staticText5.SetLabel("rouge:"+str(value[1]))
        self.loaing(False)

    def enable(self):
        self.m_bpButton1.Enable(True)
        self.m_statusBar1.SetStatusText("ready")

    def loaing(self,is_loading):
        self.m_bpButton1.Enable(not is_loading)
        self.m_statusBar1.SetStatusText("textsuming" if is_loading else "ready")
            
    def OnOpen(self,even):
        dlg=wx.FileDialog(self,"open file to load","", "*.*")
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.m_textCtrl1.SetValue(open(path,'r', encoding='UTF-8').read())
            dlg.Destroy()
    def OnAbout(self,ev):
        b=AboutDialog(self)
        web=wx.html2.WebView.New(b)
        b.GetSizer().Add(web,1, wx.ALL|wx.EXPAND, 5)
        web.SetPage(textSumApi.get_help_url(),"about")
        b.Layout()
        b.ShowModal()
            

class TextSumApi:
    @abstractmethod
    def get_html_url(self):
        pass
    
    @abstractmethod
    def OnInit(self):
        pass

    @abstractmethod
    def OnGenTextSum(self,text):
        pass
    
    @abstractmethod
    def OnSelTextSum(self,text):
        pass

    @abstractmethod
    def get_help_url(self):
        pass

app=wx.App()
textSumApi=None
def Start_GUI(api):
    global textSumApi
    textSumApi=api
    frame = sumFrame();
    frame.Show()
    init=async_run(textSumApi.OnInit,frame.enable)
    a=()
    init.run(a)
    print(threading.current_thread())
    app.MainLoop()

from app import Start_GUI
from app import TextSumApi
import time


class api(TextSumApi):
    #this function return the tensorbox website url
    def get_html_url(self):
        return "http://news.163.com/18/0123/19/D8S08G4K000189FH.html"
    #this function init the program
    def OnInit(self):
        pass
    #this function do the 生成式文本摘要,return (文本摘要,rouge)
    def OnGenTextSum(self,text):
        time.sleep(3)
        return (text+"qwe",4)
        pass
    #this function do the 摘取式文本摘要,return (文本摘要,rouge)
    def OnSelTextSum(self,text):
        time.sleep(3)
        return (text+"zxc",3)
        pass
    #this function return the help information string（html format)
    def get_help_url(self):
        return "<hrml>hello</html>"
        pass

Start_GUI(api())

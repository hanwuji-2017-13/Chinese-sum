# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Nov  6 2017)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class mainFrame
###########################################################################

class mainFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"文本摘要工具", pos = wx.DefaultPosition, size = wx.Size( -1,-1 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHints( wx.Size( 500,600 ), wx.DefaultSize )
		
		self.m_menubar2 = wx.MenuBar( 0 )
		self.m_menu2 = wx.Menu()
		self.m_menuItem3 = wx.MenuItem( self.m_menu2, wx.ID_ANY, u"加载新闻文本", wx.EmptyString, wx.ITEM_NORMAL )
		self.m_menu2.Append( self.m_menuItem3 )
		
		self.m_menubar2.Append( self.m_menu2, u"文件" ) 
		
		self.SetMenuBar( self.m_menubar2 )
		
		bSizer3 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_textCtrl7 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE )
		self.m_textCtrl7.SetMinSize( wx.Size( 600,400 ) )
		
		bSizer3.Add( self.m_textCtrl7, 0, wx.ALL|wx.EXPAND, 5 )
		
		self.m_bpButton4 = wx.BitmapButton( self, wx.ID_ANY, wx.Bitmap( u"array.bmp", wx.BITMAP_TYPE_ANY ), wx.DefaultPosition, wx.DefaultSize, wx.BU_AUTODRAW )
		bSizer3.Add( self.m_bpButton4, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		self.m_textCtrl8 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_textCtrl8.SetMinSize( wx.Size( 600,100 ) )
		
		bSizer3.Add( self.m_textCtrl8, 0, wx.ALL|wx.EXPAND, 5 )
		
		
		self.SetSizer( bSizer3 )
		self.Layout()
		bSizer3.Fit( self )
		self.m_statusBar1 = self.CreateStatusBar( 1, wx.STB_SIZEGRIP, wx.ID_ANY )
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.Bind( wx.EVT_MENU, self.OnFileOpen, id = self.m_menuItem3.GetId() )
		self.m_bpButton4.Bind( wx.EVT_BUTTON, self.OnGenerate )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def OnFileOpen( self, event ):
		event.Skip()
	
	def OnGenerate( self, event ):
		event.Skip()
	


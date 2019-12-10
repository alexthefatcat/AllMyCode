




import pandas as pd
import csv

from bs4 import BeautifulSoup
from bs4.element import Comment



#  KeepAlphaNumbericWords2
#  KeepAlphaNumbericWords
#  SaveNestedList2CSV
#  GetXpathOfSoupElement
#  TagVisible
#  VisibleTextFromHTML2
#  CreateTextListAndDfFromSoup
#  AddHomeToPath



#%%###########################################################################################################
def GetXpathOfSoupElement(element):
    # type: (typing.Union[bs4.element.Tag, bs4.element.NavigableString]) -> str
    """
    Generate xpath from BeautifulSoup4 element.
    :param element: BeautifulSoup4 element.
    :type element: bs4.element.Tag or bs4.element.NavigableString
    :return: xpath as string
    :rtype: str
    Usage
    -----
    >>> import bs4
    >>> html = (
    ...     '<html><head><title>title</title></head>'
    ...     '<body><p>p <i>1</i></p><p>p <i>2</i></p></body></html>'
    ...     )
    >>> soup = bs4.BeautifulSoup(html, 'html.parser')
    >>> xpath_soup(soup.html.body.p.i)
    '/html/body/p[1]/i'
    >>> import bs4
    >>> xml = '<doc><elm/><elm/></doc>'
    >>> soup = bs4.BeautifulSoup(xml, 'lxml-xml')
    >>> xpath_soup(soup.doc.elm.next_sibling)
    '/doc/elm[2]'
    """
    components = []
    child = element if element.name else element.parent
    for parent in child.parents:  # type: bs4.element.Tag
        siblings = parent.find_all(child.name, recursive=False)
        components.append(
            child.name if 1 == len(siblings) else '%s[%d]' % (
                child.name,
                next(i for i, s in enumerate(siblings, 1) if s is child)
                )
            )
        child = parent
    components.reverse()
    return '/%s' % '/'.join(components)

def GetElementFromSoupUsingXpath(soup, xpath):
    """    Gets you the element when given an xpath soup    """
    pars  = xpath.split("/")[1:]   
    pars2 = [f.replace("]","").split("[") for f in pars]
    pars2 = [[f[0],1] if len(f)==1 else [f[0],int(f[1])] for f in pars2]
    top = soup.findAll(pars2[0][0])
    if len(top)==0:
        pars2 = pars2[1:]
        top = soup.findAll(pars2[0][0])
    if len(top)==0:
        return None
    top = top[0]  
    celem = top # current element
    parrent_parts = pars2[1:]
    
    for i,p in enumerate(parrent_parts):
        count=1
        for sib in celem.children:       
            if sib.name==p[0]:
                if count==p[1]:
                   celem =sib
                   break
                count+=1
        else:
           return None
    #print(GetXpathOfSoupElement(celem),xpath)            
    return celem


def GetXpathOfSoupElement_IdandClass(element):
    " ids,classs = GetXpathOfSoupElement(element) "
    def GetXpathSegmentDepndentOnSibiblings(parent,attrkey,attrval):
        siblings = parent.find_all(attrs={attrkey: attrval}, recursive=False)
        return attrval if 1 == len(siblings) else '%s[%d]' % (attrval, next(i for i, s in enumerate(siblings, 1) if s is child))     
    componentsid,componentscl = [],[]
    child = element if element.name else element.parent
    for parent in child.parents:  # type: bs4.element.Tag
        componentsid.append(GetXpathSegmentDepndentOnSibiblings(parent, "id"   , child.get("id","")               ))        
        componentscl.append(GetXpathSegmentDepndentOnSibiblings(parent, "class", " ".join(child.get("class",[]))  ))
        child = parent
    componentsid.reverse()
    componentscl.reverse()  
    return '/'.join(componentsid) , '/'.join(componentscl) 



def TagVisible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def VisibleTextFromHTML2(body,elements=False):
    if type(body) is str:
        soup = BeautifulSoup(body, 'html.parser')
    else:
        soup = body
    texts = soup.findAll(text=True)
    visible_texts = list(filter(TagVisible, texts))
    if elements:
        return visible_texts
    return u" ".join(t.strip() for t in visible_texts)
    # "\t".join(t.strip().replace("\t","    ") for t in visible_texts)




def CreateTextListAndDfFromSoup(soup,only_xpaths=False,return_text=False):
    #vtext0           = VisibleTextFromHTML2(html,True)
    vtext0           = VisibleTextFromHTML2(soup,True)  
    #locations        = [i for i,t in enumerate(vtext0) if len(t.strip())>0]
    vtexts           = [i for i in vtext0 if len(i.strip())>0]
    texts            = [str(i) for i in vtexts]
    
    xpaths           = [GetXpathOfSoupElement(i) for i in vtexts]
    #vtexts           = [vtext0[i] for i in locations]
    #textsparents     = [i.parent for i in vtexts] 
    if only_xpaths:
        if return_text:
            return texts, xpaths
        return vtexts, xpaths
   
    columns = ["length","link","font","fontsize","italic","bold","xpath","link_parents","org_loc","header","group"]
    
    texts_df         = pd.DataFrame(index=range(len(vtexts)),columns=columns).fillna("")
    
    texts_df["length"        ] = [len(i) for i in vtexts] 
    texts_df["link"          ] = [text.parent.get("href","")  for text in vtexts]
    texts_df["class_parrent" ] = [" ".join(text.parent.get("class",[]))  for text in vtexts]
    texts_df["class_parrent2"] = [" ".join(text.parent.parent.get("class",[]))  for text in vtexts]
    
    texts_df["class_parrents"] = ["*".join([  " ".join(par.get("class",[])) for par in text.parents] )  for text in vtexts]

    texts_df["link_parents"  ] = [" ".join([n for n in [n.get("href","") for n in vtext0_.parents] if not n ==""]) for vtext0_ in vtexts]
    texts_df["org_loc"       ] = list(range(len(vtexts)))#locations   
    texts_df["xpath"         ] = xpaths
    if return_text:
       return  texts, texts_df        
    return vtexts,texts_df



#%%###########################################################################################################
   
    


def GetStyleInfoForBeautifulSoupElement(browser,bs_elem=False,xpath=False,return_xpath=False):
    if xpath==False:
       xpath    = GetXpathOfSoupElement(bs_elem)
    elements = browser.find_element_by_xpath(xpath)   
    #out = {k:element.value_of_css_property(k) for k in "font-size font-style font-weight height".split(" ")}
    style_dict = {k:elements.value_of_css_property(k) for k in "font-size font-style font-weight height font-family".split(" ")}    
    if return_xpath:
        return style_dict,xpath
    return style_dict

def FindStyleInformationAndAddToDataFrame(browser,dataframe=None,locationinfo=False,tryify=False):
    """Based ib the xpath column it adds four new ones with style information about this element   """
    
    columns_dict = {"italic"  :"font-style",
                    "fontsize":"font-size",
                    "bold"    :"font-weight",
                    "font"    :"font-family"}
    columns2 = [[],["locx","locy","sizx","sizy"]][locationinfo]
                
    if not dataframe is None:
        for extra_col in list(columns_dict)+columns2:
            if extra_col not in dataframe.columns:
               dataframe[extra_col] = None
        
        for i in dataframe.index:
            if i%20==(20-1):
               print(f"Finding CCS info for element:{str(i+1).rjust(5)} / {len(dataframe.index)}")
            xpath =  dataframe.loc[i,"xpath"]
            if tryify:
                try:
                    style_dict = GetStyleInfoForBeautifulSoupElement(browser,xpath=xpath)              
                    for kcol,vsty in columns_dict.items():
                        dataframe.loc[i, kcol ] = style_dict[ vsty ]
                    if locationinfo:
                        ele = browser.find_element_by_xpath(xpath) 
                        dataframe.loc[i, columns2 ] = list(ele.location.values())+list(ele.size.values())
                except:
                    pass
            else:
                style_dict = GetStyleInfoForBeautifulSoupElement(browser,xpath=xpath)              
                for kcol,vsty in columns_dict.items():
                    dataframe.loc[i, kcol ] = style_dict[ vsty ]
                if locationinfo:
                    ele = browser.find_element_by_xpath(xpath) 
                    dataframe.loc[i, columns2 ] = list(ele.location.values())+list(ele.size.values())
        return dataframe




def FindLocationOfElementSelenium(browser):
    """ This needs some work
    """
    browser.maximize_window() # now screen top-left corner == browser top-left corner 
    browser.get("http://stackoverflow.com/questions")
    question = browser.find_element_by_link_text("Questions")
    y_relative_coord = question.location['y']
    browser_navigation_panel_height = browser.execute_script('return window.outerHeight - window.innerHeight;')
    y_absolute_coord = y_relative_coord + browser_navigation_panel_height
    x_absolute_coord = question.location['x']
    return x_absolute_coord,y_absolute_coord



#%%###########################################################################################################

def AddHomeToPath(home,child):
    if not child.startswith("/"):
        return child
    #print(home.split("/")[:3]+[child])
    return "/".join( home.split("/")[:3])+child



def FindXpathOfSegmentsFromXpathOfTitles(xpaths):
    """Find the xpath of the segment which contains the title """
    def KeepStart(a,b):
         loc = [i for i,(aa,bb) in enumerate(zip(a,b)) if aa!=bb][0]
         return "/".join(b[:loc+1])    
    vv   = [n.split("/") for n in xpaths]
    vvv  = [[ KeepStart(a,b) for a in vv if a!=b ] for b in vv]
    vvvv = [max(b, key=lambda s: len(s)) for b in vvv]
    return vvvv

def GetHighestRootOfXpathWhichIsUnique(xpaths):
    xpathsshift      = [m.split("/") for m in xpaths ]
    xpathsuniqueroot = [  "" for m in xpathsshift]   
    while not all([n==[] for n in xpathsshift]):
        xpathsuniqueroot = [  b if xpathsuniqueroot.count(b)==1 else b+"/"+a[0] for a,b in zip(xpathsshift,xpathsuniqueroot)]
        xpathsshift      = [ [] if xpathsuniqueroot.count(b)==1 else      a[1:] for a,b in zip(xpathsshift,xpathsuniqueroot)]
    xpathsuniqueroot = [x[1:] for x in xpathsuniqueroot]    
    return xpathsuniqueroot

def Create_GetDateFromContent():   
    # maybe seperate numbers and text,      # also format //
    # remove none alpha but replace with space only one space
    def RemoveNoneAlphaNumericFromString(string,replace=" "):
        out = [ s if s in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" else replace for s in string ]
        out = "".join(out).split(" ")
        out = [w for w in out if w !=""]
        return " ".join(out)
    
    def ConvertToWords(string):
        string = RemoveNoneAlphaNumericFromString(string)
        words = string.lower().split()
        return words
    
    Months      ="January/February/March/April/May/June/July/August/September/October/November/December".lower().split("/")
    MonthsShort = [m[:3] for m in Months]
    days        = '1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th 11th 12th 13th 14th 15th 16th 17th 18th 19th 20th 21st 22nd 23rd 24th 25th 26th 27th 28th 29th 30th 31st'.split()
    years       = [str(n) for n in range(2009,2031)]
    daysn       = [str(n) for n in range(1,32)]
    
    def IsGetDay(string):
       string = string.lstrip("0")
       if   string in days:
            return (True, days.index(string)+1)
       elif string in daysn:
            return (True, daysn.index(string)+1)
       return (False,None)
       
    def Get_MonthsLocation(words):
        monthlocs = [i for i,w in enumerate(words) if any([m==w for m in Months])]
        return monthlocs
    
    def Get_Months2Location(words):
        monthlocs = [i for i,w in enumerate(words) if any([m==w for m in MonthsShort])]
        return monthlocs
    
    def GetNeibouringWords(words,func,n=2):
        words2 = ([""]*n)+words+([""]*n)
        locs = func(words2)
        return [ words2[(loc-n):(loc+n+1)] for loc in locs]      
    
    def FindDate(posdatestring):# given 5 words
        if len(posdatestring)==5:
            #   Month
            yea_mon_day    = [None,None,None]
            if   posdatestring[2] in Months:
               yea_mon_day[1] = Months.index(posdatestring[2])+1
            elif posdatestring[2] in MonthsShort:
               yea_mon_day[1] = MonthsShort.index(posdatestring[2])+1        
            else:
                return None
            del posdatestring[2]
            #   Years        
            if   posdatestring[3] in years:
               yea_mon_day[0] = int(posdatestring[3])
               del posdatestring[3]
            elif posdatestring[2] in years:
               yea_mon_day[0] = int(posdatestring[2])
               del posdatestring[2]            
            else:
                return None
            # Day of the week    
            dayflag1,day1_ = IsGetDay(posdatestring[1])
            dayflag2,day2_ = IsGetDay(posdatestring[2])
            if   dayflag1:
                del posdatestring[1]
                yea_mon_day[2] = day1_
            elif dayflag2:
                del posdatestring[2]
                yea_mon_day[2] = day2_
                return yea_mon_day 
            else :
                return None
            return yea_mon_day   
        return None

    def GetDateFromSlashFormat(content):
        try:
            words = [n for n in content.split() if "/" in  n]
            if len(words)>0:
                words = [w.split("/") for w in words ]
                words = [w for w in words if len(w)==3] 
                words = [w for w in words if all([n.isdigit() for n in w])]     
                words = [w for w in words if (32>int(w[0])>0)and(13>int(w[1])>0)and(2031>int(w[2])>2012)  ]
                if len(words)>0:
                    word = words[0]
                return [int(word[2]),int(word[1]),int(word[0])]
        except:
            pass
        return "" 
    
    def GetDateFromContent(content,singleflag=False):  
        if   type(content) is str:
            words = ConvertToWords(content)
        elif type(content) is list:
            words = content 
        else:
            return None
        possibledatewords5 = GetNeibouringWords(words,Get_MonthsLocation)
        possibledatewords5_ = [FindDate(n) for n in possibledatewords5]  
        possibledatewords5_ = [n for n in possibledatewords5_ if n not in ([],None)]
        if len(possibledatewords5_)>0:
            if singleflag:
                return possibledatewords5_[0],False                
            return possibledatewords5_[0]
        
        possibledatewords5 = GetNeibouringWords(words,Get_Months2Location)
        possibledatewords5_ = [FindDate(n) for n in possibledatewords5]  
        possibledatewords5_ = [n for n in possibledatewords5_ if n not in ([],None)]
        if len(possibledatewords5_)>0:
            if singleflag:
                return possibledatewords5_[0],False              
            return possibledatewords5_[0]     
        
        # try getting it from slash format
        if singleflag:
            return GetDateFromSlashFormat(content),True 
        return GetDateFromSlashFormat(content)
    return GetDateFromContent

#GetDateFromContent("February 25, 2018 By Janin")
#content = "February 25, 2018 By Janin"
#content = "and then a dog eat his sister she wa not pleasred February 25, 2018 By Janin ham ham ham it is nice"
#
#GetDateFromContent(content)







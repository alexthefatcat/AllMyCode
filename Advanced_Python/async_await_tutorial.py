# -*- coding: utf-8 -*-
"""Created on Mon Dec  9 11:03:00 2019@author: Alexm"""
###########################################################################
#
#            Asynchronous Programming in Python
#
#    cocurrency can do other tasks while waiting on another one
#    not multiprocessing as only actively dong one task at a time
#    coroutines are functions whose execution you can pause. 
#    similar to generators
#
#
#     create a loop with asyncio.get_event_loop()
#     loop.run_until_complete(async_func)
#     loop.close()    close if 
#
#     for the async function
#     put async in front of it 
#     await asyncio.wait to await a lost of tasks
#     asynico.sleep  to sleep
#   
#
# 
#   async def asynce_func(a,b): 
#         asynico.sleep(0.3)  
#         return a+b  
#
#   async def main():
#       task1 = loop.create_task( 10,9)
#       task2 = loop.create_task( 30,9)
#       await asyncio.wait([task1, task2 ])
#       return task1, task2
#          
#   loop = asyncio.get_event_loop()
#   loop.run_until_complete(async_func)
#
#
############################################################################

mode = ("sync","async")[1]
print(mode)

if mode =="sync":
    def find_divisibles(inrange, div_by):
        print("finding nums in range {} divisible by {}".format(inrange, div_by))
        located = []
        for i in range(inrange):
            if i % div_by == 0:
                located.append(i)
        print("Done w/ nums in range {} divisible by {}".format(inrange, div_by))
        return located
    def main():
        divs1 = find_divisibles(508000, 34113)
        divs2 = find_divisibles(100052, 3210)
        divs3 = find_divisibles(500, 3)
    if __name__ == '__main__':
        main()

########################################################################################

if mode =="async":

    import asyncio
    ## Function to Workon
    async def find_divisibilites(inrange,div_by,name=""):
        print(f"Finding nums in range {inrange}, divisible by {div_by}")
        
        located =[]
        for i in range(inrange):
            if i % div_by ==0:
                located.append(i)
            if i % 50_000 ==0 :
                print(f"      >>Sleep  {name}")
                await asyncio.sleep(0.001)
        print("  >> Finished")
        return  located     
         
    async def main():
        divs1 = loop.create_task( find_divisibilites(500_000,34,"big"   ))
        divs2 = loop.create_task( find_divisibilites(5000   ,33,"medium"))
        divs3 = loop.create_task( find_divisibilites(500    ,31,"small" ))
        await asyncio.wait([divs1,divs2,divs3])
        return divs1, divs2, divs3 
    
    if __name__=="__main__":
        loop = asyncio.get_event_loop()
        loop.set_debug(1)   # useful for debuggin
        d1,d2,d3 = loop.run_until_complete(main())
        print(d1.result())
        loop.close()    
        print("Finished**")

   
   
# -*- coding: utf-8 -*-
"""Created on Mon Feb  3 23:19:02 2020@author: Alexm"""




def print_swap_array(array,i,pivot,end=False):
    array2 = [a for a in array]
    array2[i    ] = (array2[i],)
    array2[pivot] = (array2[pivot],)
    if end:
       print(">",array2,"\n")
    else:
       print(array2)    



def QuickSort(array, begin=0, end=None):
    """ The partitions first element is the pivot value
    The location the pivot is going to be placed at the end is i
    The ones smaller than pivot are moved to before i and i++
    """
    def swap(array,a,b):
        array[a], array[b] = array[b], array[a]
        return array
    
    def partition(array, begin, end):
        pivot = begin # pivot is the location of the pivot value
        for i in range(begin+1, end+1): # all locations apart from the firest one
            if array[i] <= array[begin]:
                pivot += 1
                print_swap_array(array,i,pivot)##<<
                array = swap(array,i,pivot)
        print_swap_array(array,pivot,begin,True)##<<
        array = swap(array,pivot,begin)
        return pivot
    
    if end is None:
        end = len(array) - 1
    if begin >= end:
        return array
    pivot = partition(array, begin, end)
    array = QuickSort(array, begin, pivot-1)
    array = QuickSort(array, pivot+1, end)
    return array

def BubbleSort(lis):
   """Slow 
   """
   llen = len(lis)
   while True:
      llen -=1
      out = True
      for i,e in range(llen):
         if lis[i]>lis[i+1]:
            lis[i],lis[i+1] = lis[i+1],lis[i] 
            out = False
      if out:
         return lis




if __name__ == "__main__":
    
    print("    Quick Sort Algorithum")
    array = [6,5,8,4,3,7,2,1,98,4,55,214,45,3,4,55]
    QuickSort(array)
    print("Sorted Array",array)
    
    
    
    
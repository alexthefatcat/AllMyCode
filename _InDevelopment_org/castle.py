
def castle_1(x,y):   
    print("--------------------------------------------------")    
    print("%s approaches an old, mysterious castle, to rescue the princess..." % (x))    
    print("You've entered the castle...")
    print("It's very spooky...")
    print("You can see doors, decorated with the numbers 1 to 4...")    
    def castle_2():
        answer_1 = input("Choose a door to enter...or run away...")
        if answer_1 == "1":
            print("--------------------------------------------------")
            print("You venture forward...and fall into the dungeon!")
            answer_2 = input("A troll called %s emerges from a dark corner of the dungeon.  Will you fight it?" % (y)).lower() 
            if answer_2 == "y" or answer_2 == "yes":
                print("--------------------------------------------------")
                print("You try to fight, but the troll eats you!")
            elif answer_2 == 'n' or answer_2 == 'no':
                print("--------------------------------------------------")
                print("You avoid the troll, and manage to climb out of the dungeon.")
                castle_2()
            else:    
                print("--------------------------------------------------")
                print("The troll decides to keep you as his pet!") 
        elif answer_1 == "2":
            print("--------------------------------------------------")            
            print("You've found the princess...you rescue her and escape!")                
        elif answer_1 == "3":
            print("--------------------------------------------------")            
            print("You've found the treasure!  Now rescue the princess!")
            castle_2()
        elif answer_1 == "4":
            print("--------------------------------------------------")
            print("You are crushed by a troll!")                
        else:
            print("--------------------------------------------------")
            print("You run away...but as you cross the drawbridge "\
            "a dragon from the moat chases you back inside!")
            castle_2()
    castle_2()

   
castle_1("Name_1", "Name_2")



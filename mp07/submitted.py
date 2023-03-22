'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    standardized_rules = copy.deepcopy(nonstandard_rules)
    variables = []
    title = "x000"
    i = 1
    for key in standardized_rules: # key is the key of given dict
    # key may not be rule! can also be triple
        if key[0]=="r": 
            new_name = title + str(i)
            for antecedent in standardized_rules[key]['antecedents']:
                antecedent[0] = new_name
            # standardized_rules[rule]['antecedents'][0][0] = new_name
            standardized_rules[key]['consequent'][0] = new_name
            variables+=[new_name]
            i += 1
    
    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples: skipped
    [ A, needs, B, True ]
    '''
    # Write a lot of if statement 12 is good
    unification = [] 
    subs = {} # substitution
    q = copy.deepcopy(query)
    d = copy.deepcopy(datum)
    
    if(q == None or d == None) or (q[3] ^ d[3] == True) or (q[1] != d[1]):
        return None,None

    for i in [0,2]: 
        if q[i] in variables:
            subs[q[i]] = d[i] # When the 'x':'a' substitution is detected.
        else:
            subs[d[i]] = q[i] # else q[0] is a symbol
        for j in [0,2]:
            if q[j] in subs.keys(): # duplicate case
                q[j] = subs[q[j]]   # repoint to the datum
    unification = q[:]  # turn back
    return unification, subs
'''
        
    if(q[0] in variables ):
        unification[0] = (d[0])

        # if d[0] in variables:
        #     subs[q[0]] = d[0] 
        # else:
        subs[q[0]] = d[0] 
    else:
        unification[0] = (q[0])
        subs[d[0]] = q[0]
    unification[1] = q[1]

    if(q[2] in variables):
        unification[2] = (d[2])
        subs[q[2]] = d[2]
    else:
        unification[2] = (q[2])
        subs[d[2]] = q[2]
    unification[3] = q[3]
    if(q[0] in variables and q[0] == q[2]):
        unification[0] = (d[0])
        unification[2] = (d[0])

'''





def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Examples: skipped
    '''
    applications = []
    goalsets = []
    G = copy.deepcopy(goals)
    # i = 0
    for goal in G:
        # print(i,'\n----')
        unified, subs = unify(rule['consequent'], goal, variables)
        if unified is not None:
            index = G.index(goal)
            new_goals = G[:index] + G[index+1:] # 
            antecedents = []
            # print("unified ",i," is: " ,unified)
            for proposition in rule['antecedents']:
                new_state = []
                for word in proposition:
                    if word in subs.keys():
                        word = subs[word]
                    new_state.append(word)
                new_state[0] = unified[0] # debug, why the first don't output?
                antecedents.append(new_state)

            applications.append({'antecedents':antecedents, 'consequent':unified})
            goalsets.append(new_goals + antecedents)
        # i += 1
        
    return applications, goalsets





def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''

    proof = []
    V = list(variables)
    for rule in rules.values():
        print(rule) 
        applied, goalset = apply(rule,query,variables)
        print("applied is ", applied)
        print("goalset is ", goalset )
        break
    print(query)
    print(variables)

    

    return proof

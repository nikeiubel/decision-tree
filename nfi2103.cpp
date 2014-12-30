//
//  nfi2103.cpp
//  Decision Tree Learning
//
//  By Nikolas Iubel.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <unistd.h>

using namespace std;

//These structs describe a tree in terms of its nodes and arcs (edges)
struct decisionTreeNode {
    int attribute;
    string label;
    vector< decisionTreeNode> children;
};

struct decisionTreeArc {
    decisionTreeNode parent;
    decisionTreeNode child;
    string label;
};

void StringToVector (string str, vector<string> & vec);
void PrintExamples (vector< vector<string> > & examples);
string ClearSpacesBefore (string str);
string ClearSpacesAfter (string str);
void CreateAttributeVector (vector< int> & attributes, int numAttributes);
void DecisionTreeLearning (vector< vector<string> > & examples, vector< int> & attributes, vector< vector<string> > & parent_examples, decisionTreeNode & tree, vector< decisionTreeArc> & arcs, int numAttributes, vector< vector<string> > & allExamples);
void PrintVectorStr (vector< string> & vec);
string PluralityValue (vector< vector<string> > & examples, int numAttributes);
bool AllSameClassification (vector< vector<string> > & examples, int numAttributes);
int MaxImportance (vector< vector<string> > & examples, vector< int> attributes, int numAttributes);
double Entropy (map<string, double> & labelAndCount);
double CalcRemainder (vector< vector<string> > & examples, int attribute, int numAttributes);
void CopyEntriesWithSameAttr (vector< vector<string> > & examples, vector< vector<string> > & examplesWithThisAttr, int attribute, string attr);
void CountLabels (map<string, double> & labelAndCount, vector< vector<string> > & examples, int numAttributes);
bool NoAttributesLeft (vector< int> & attributes);
void RearrangeArcs (decisionTreeNode & tree, vector< decisionTreeArc> & arcs, vector< decisionTreeArc> & reorderedArcs);
void PrintTree (decisionTreeNode & tree);
void PrintArcs (vector< decisionTreeArc> & arcs);
void OutputProgram (vector< decisionTreeArc> & arcs);

//void AssignLabels (vector< decisionTreeArc> & arcs, vector< vector<string> > & examples);
//void PrintExampleRow (vector<string> & exampleRow, string label);


int main(int argc, const char * argv[])
{

    // Initializes variables
    string filename;
    int numAttributes;
    vector< vector<string> > examples, parent_examples;
    vector< int> attributes;
    ifstream infile;
    
    //reads name of training file from user input
    while (true) {
        cout << "Enter name of training data file:" << endl;
        getline (cin, filename);
        //infile.open("/Users/nikeiubel/Desktop/Artificial Intelligence/restaurant2_train.csv");
        infile.open(filename.c_str());
        if (infile.fail()) cout << "Could not open this file. Try again." << endl;
        else break;
        infile.clear();
    }
    
    //attempts to open file whose name user entered as input
    while (true) {
        string exampleStr;
        vector<string> exampleVec;
        getline(infile,exampleStr);
        if (infile.fail()) break;
        StringToVector(exampleStr, exampleVec);
        examples.push_back(exampleVec);
    }
    
    numAttributes = (int)examples.at(0).size() - 1;
    
    //attributes are represented by their position in the training dataset
    //they are therefore integers ranging from 0 to numAttributes-1
    CreateAttributeVector(attributes, numAttributes);
    
    decisionTreeNode tree;
    tree.attribute = -1;
    tree.label = "";
    tree.children.clear();
    
    vector< decisionTreeArc> arcs;
    arcs.clear();
    
    DecisionTreeLearning (examples, attributes, parent_examples, tree, arcs, numAttributes, examples);

    //makes sure arcs are stored in a way that preserves the sequence of tree nodes
    vector< decisionTreeArc> reorderedArcs;
    reorderedArcs.clear();
    RearrangeArcs(tree, arcs, reorderedArcs);
    
    infile.close();
    
    OutputProgram (reorderedArcs);
    
    return 0;
}

//cleans up input data by getting rid of white spaces, and
//converts a string representing an example to a vector of strings
void StringToVector (string str, vector<string> & vec) {
    int pos = -1;
    do {
        pos = (int)str.find(',');
        string token = str.substr(0,pos);
        token = ClearSpacesBefore(token);
        token = ClearSpacesAfter(token);
        vec.push_back(token);
        str = str.substr(pos+1);
    } while (pos >= 0);
}

//used to debug, allows me to see content of vector of vectors of strings 'examples'
void PrintExamples (vector< vector<string> > & examples) {
    for (vector< vector<string> >::iterator it1 = examples.begin(); it1 != examples.end(); ++it1) {
        cout << "An example is: ";
        for (vector<string>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
            cout << *it2 << " ";
        }
        cout << endl;
    }
}

string ClearSpacesBefore (string str) {
    while (iswspace(str[0])) str = str.substr(1);
    return str;
}

string ClearSpacesAfter (string str) {
    while (iswspace(str[str.length()-1])) str = str.substr(0,str.length()-1);
    return str;
}

//attributes are represented by their position in the training dataset
//they are therefore integers ranging from 0 to numAttributes-1
void CreateAttributeVector (vector< int> & attributes, int numAttributes) {
    for (int i = 0; i < numAttributes; i++) {
        attributes.push_back(i);
    }
}

//used to debug, prints contents of a vector of strings which stores attributes of one example
void PrintVectorStr (vector< string> & vec) {
    for (vector< string>::iterator it3 = vec.begin(); it3 != vec.end(); ++it3) {
        cout << "Attribute is: " << *it3 << " " << endl;;
    }
}

//Recursive Function whose pseudo-code is provided in Chapter 18, Figure 18.5 of textbook
void DecisionTreeLearning (vector< vector<string> > & examples, vector< int> & attributes, vector< vector<string> > & parent_examples, decisionTreeNode & tree, vector< decisionTreeArc> & arcs, int numAttributes, vector< vector<string> > & allExamples) {
    int A;
    if (examples.empty()) {
        tree.attribute = -1;   //attribute = -1 indicates absence of further attributes to decide on
        tree.label = PluralityValue (parent_examples, numAttributes);   // label indicates decision output
        tree.children.clear();
        return;
    }
    else if (AllSameClassification(examples, numAttributes)) {
        tree.attribute = -1;
        tree.label = examples.at(0).at(numAttributes);
        tree.children.clear();
        return;
    }
    else if (NoAttributesLeft(attributes)) {
        tree.attribute = -1;
        tree.label = PluralityValue(examples, numAttributes);
        tree.children.clear();
        return;
    }
    else {
        A = MaxImportance(examples, attributes, numAttributes);
        tree.attribute = A;
        map<string, double> attrOfA;
        CountLabels(attrOfA, allExamples, A);
        for (map <string, double>::iterator it = attrOfA.begin(); it != attrOfA.end(); ++it) {
            vector< vector<string> > exs;
            CopyEntriesWithSameAttr (examples, exs, A, it->first);
            attributes.at(A) = -1;    //remove attribute from consideration from now on
            decisionTreeNode subtree;
            DecisionTreeLearning(exs, attributes, examples, subtree, arcs, numAttributes, allExamples);
            tree.children.push_back(subtree);
            tree.label = "";
            
            //keeps track of which label connects each pair of parent and child nodes
            decisionTreeArc arc;
            arc.parent = tree;
            arc.child = subtree;
            arc.label = it->first;
            arcs.push_back(arc);

        }
    }
    return;
}

//selects the most common output among a set of examples, breaking ties arbitrarily
string PluralityValue (vector< vector<string> > & examples, int numAttributes) {
    map <string, double> labelAndCount;
    CountLabels(labelAndCount, examples, numAttributes);
    int maxValue = 0;
    string maxStr;
    for (map <string, double>::iterator it4 = labelAndCount.begin(); it4 != labelAndCount.end(); ++it4) {
        if (it4->second > maxValue) {
            maxValue = it4->second;
            maxStr = it4->first;
        }
    }
    return maxStr;
}

//checks whether all examples in the vector of vector of strings 'examples' have same classification
bool AllSameClassification (vector< vector<string> > & examples, int numAttributes) {
    string classification = examples.at(0).at(numAttributes);
    for (int i=0; i < examples.size(); i++) {
        if (examples.at(i).at(numAttributes) != classification) return false;
    }
    return true;
}

//returns attribute that offers the most information gain among the ones not used it
//this attribute is therefore the best attribute to split the decision Tree on
int MaxImportance (vector< vector<string> > & examples, vector< int> attributes, int numAttributes) {
    map <string, double> labelAndCount;
    CountLabels(labelAndCount, examples, numAttributes);
    double B = Entropy(labelAndCount);
    double maxInfoGain = 0;
    int maxInfoGainAttr = -1;
    for (vector < int>::iterator it = attributes.begin(); it != attributes.end(); ++it) {
        if (*it >= 0) {
            double infoGain = B - CalcRemainder(examples, *it, numAttributes);
            if (infoGain > maxInfoGain) {
                maxInfoGain = infoGain;
                maxInfoGainAttr = *it;
            }
        }
    }
    //most of the time, this is when MaxImportance returns 
    //(will return here when there is some attribute offering positive gain)
    if (maxInfoGainAttr >= 0) return maxInfoGainAttr;
    
    //when all attributes provide zero gain, return the first one not used yet
    for (vector < int>::iterator it2 = attributes.begin(); it2 != attributes.end(); ++it2) {
        if (*it2 >= 0) return *it2;
    }
    
    //style purposes: this line should never get executed
    return 0;
}

//takes as argument a map with key, value pairs describing the frequency with which a given
//attribute label occurs among a set of examples
double Entropy (map<string, double> & labelAndCount) {
    vector< double> labelCounts;
    for (map <string, double>::iterator it1 = labelAndCount.begin(); it1 != labelAndCount.end(); ++it1) {
        labelCounts.push_back(it1->second);
    }
    double totalCount = 0;
    for (vector< double>::iterator it2 = labelCounts.begin(); it2 != labelCounts.end(); ++it2) {
        totalCount += *it2;
    }
    for (vector< double>::iterator it3 = labelCounts.begin(); it3 != labelCounts.end(); ++it3) {
        *it3 = *it3/totalCount;
        *it3 *= log2(*it3);
    }
    double sum = 0;
    for (vector< double>::iterator it4 = labelCounts.begin(); it4 != labelCounts.end(); ++it4) {
        sum += *it4;
    }
    return (-1)*sum;
}

//calculates expected entropy remaining after testing a given attribute
//this function implements REMAINDER as described in section 18.3.4 of the textbook
double CalcRemainder (vector< vector<string> > & examples, int attribute, int numAttributes) {
    map <string, double> attrAndCount;
    CountLabels(attrAndCount, examples, attribute);
    double totalCount = 0;
    for (map <string, double>::iterator it1 = attrAndCount.begin(); it1 != attrAndCount.end(); ++it1) {
        totalCount += it1->second;
    }
    for (map <string, double>::iterator it2 = attrAndCount.begin(); it2 != attrAndCount.end(); ++it2) {
        it2->second /= totalCount;
        vector< vector<string> > examplesWithThisAttr;
        CopyEntriesWithSameAttr(examples,examplesWithThisAttr, attribute, it2->first);
        map <string,double> submapWithThisAttr;
        CountLabels(submapWithThisAttr, examplesWithThisAttr, numAttributes);
        double B = Entropy(submapWithThisAttr);
        it2->second *= B;
    }
    double totalRemainder = 0;
    for (map <string, double>::iterator it3 = attrAndCount.begin(); it3 != attrAndCount.end(); ++it3) {
        totalRemainder += it3->second;
    }
    return totalRemainder;
}

//this function returns the subset of examples entries (rows of the example table) that share
//the same attribute label for a given attribute
void CopyEntriesWithSameAttr (vector< vector<string> > & examples, vector< vector<string> > & examplesWithThisAttr, int attribute, string attr) {
    for (vector< vector<string> >::iterator it1 = examples.begin(); it1 != examples.end(); ++it1) {
        if (it1->at(attribute) == attr) examplesWithThisAttr.push_back(*it1);
    }
}

//calculates the frequency with which a given attribute label occurs among a set of examples
//and stores it in a map
void CountLabels (map<string, double> & labelAndCount, vector< vector<string> > & examples, int attribute) {
    for (int i=0; i < examples.size(); i++) {
        if (labelAndCount.count(examples.at(i).at(attribute)) == 0) labelAndCount.insert(pair<string,int> (examples.at(i).at(attribute),1));
        else
            labelAndCount.find(examples.at(i).at(attribute))->second++;
    }
}

//checks if there are further attributes to consider spliting the tree on 
bool NoAttributesLeft (vector< int> & attributes) {
    for (vector< int>::iterator it = attributes.begin(); it != attributes.end(); ++it) {
        if (*it != -1) return false;
        
    }
    return true;
}

//makes sure arcs are stored in a way that preserves the sequence of tree nodes
void RearrangeArcs (decisionTreeNode & tree, vector< decisionTreeArc> & arcs, vector< decisionTreeArc> & reorderedArcs) {
    
    for (vector <decisionTreeArc>::iterator it1 = arcs.begin(); it1 != arcs.end(); ++it1) {
        if (it1->parent.attribute == tree.attribute) reorderedArcs.push_back(*it1);
    }
    
    if (reorderedArcs.size() >= arcs.size()) return;
    
    for (vector < decisionTreeNode>::iterator it2 = tree.children.begin(); it2 != tree.children.end(); ++it2) {
        RearrangeArcs(*it2,arcs,reorderedArcs);
    }
 
    return;
}

//used to debug, prints contents of tree nodes
void PrintTree (decisionTreeNode & tree) {
    cout << "tree.attribute is " << tree.attribute << endl;
    if (tree.label.length() != 0) cout << "tree.label is " << tree.label << endl;
    else cout << "tree.label is empty" << endl;
    if (tree.children.empty()) return;
    else {
        for (vector < decisionTreeNode>::iterator it = tree.children.begin(); it != tree.children.end(); ++it) {
            PrintTree(*it);
        }
    }
}

//used to debug, prints contents of tree arcs
void PrintArcs (vector< decisionTreeArc> & arcs) {
    for (vector <decisionTreeArc>::iterator it = arcs.begin(); it != arcs.end(); ++it) {
        if (it->parent.attribute >= 0) cout << "PARENT: parent.attribute is " << it->parent.attribute << endl;
        if (it->parent.label.length() != 0) cout << "PARENT: parent.label is " << it->parent.label << endl;
        if (it->child.attribute >= 0) cout << "CHILD: child.attribute is " << it->child.attribute << endl;
        if (it->child.label.length() != 0) cout << "CHILD: child.label is " << it->child.label << endl;
        cout << "it->label is " << it->label << endl;
    }
}

//outputs a .cpp file capable of labeling examples of a test dataset
//based on the decision Tree constructed above
void OutputProgram (vector< decisionTreeArc> & arcs) {
    
    ofstream outfile;
    //outfile.open("/Users/nikeiubel/Desktop/Artificial Intelligence/decisionTree.cpp");
    outfile.open("nfi2103-decisionTree.cpp");
    outfile << "#include <iostream> \n";
    outfile << "#include <fstream> \n";
    outfile << "#include <sstream> \n";
    outfile << "#include <stdio.h> \n";
    outfile << "#include <string> \n";
    outfile << "#include <vector> \n";
    outfile << "#include <map> \n";
    outfile << "#include <math.h> \n";
    
    outfile << "using namespace std; \n";
    
    outfile << "void AssignLabels (vector< vector<string> > & examples); \n";
    outfile << "void PrintExampleRow (vector< string> & exampleRow, string label); \n";
    outfile << "void StringToVector (string str, vector<string> & vec); \n";
    outfile << "string ClearSpacesBefore (string str); \n";
    outfile << "string ClearSpacesAfter (string str); \n";
    
    outfile << "int main(int argc, const char * argv[]) { \n";
    
    outfile << "string filename; \n";
    outfile << "int numAttributes; \n";
    outfile << "vector< vector<string> > examples; \n";
    outfile <<  "vector< int> attributes; \n";
    outfile << "ifstream infile; \n";
    
    outfile << "while (true) { \n";
    outfile << "cout << \"Enter name of test data file:\" << endl; \n";
    outfile << "getline (cin, filename); \n";
    outfile << "infile.open(filename.c_str()); \n";
    outfile << "if (infile.fail()) cout << \"Could not open this file. Try again.\" << endl; \n";
    outfile << "else break; \n";
    outfile << "infile.clear(); \n";
    outfile << "} \n";
    
    outfile << "while (true) { \n";
    outfile << "string exampleStr; \n";
    outfile << "vector<string> exampleVec; \n";
    outfile << "getline(infile,exampleStr); \n";
    outfile << "if (infile.fail()) break; \n";
    outfile << "StringToVector(exampleStr, exampleVec); \n";
    outfile << "examples.push_back(exampleVec); \n";
    outfile << "} \n";
   
    outfile << "AssignLabels(examples); \n";
    outfile << "} \n";
    
    vector< decisionTreeArc> arcsCopy = arcs;
    outfile << "void AssignLabels (vector< vector<string> > & examples) { \n";
    outfile << "bool yetToPrint =  true; \n";
    outfile << "int child = -1; \n";
    outfile << "bool firstPass = true; \n";
    outfile << "for (vector< vector<string> >::iterator it1 = examples.begin(); it1 != examples.end(); ++it1) { \n";
    
    int oldAttribute = -1;
    for (vector <decisionTreeArc>::iterator it2 = arcs.begin(); it2 != arcs.end(); ++it2) {
        if (it2->parent.attribute != oldAttribute) {
            outfile << "if (firstPass) { \n";
                outfile << "child = " << it2->parent.attribute << "; \n";
                outfile << "firstPass = false; \n";
            outfile << "} \n";
            outfile << "if ((it1->at(" << it2->parent.attribute << ") == \"" << it2->label << "\") && yetToPrint && (child == " << it2->parent.attribute << ")) { \n";
            }
        else
            outfile << "else if ((it1->at(" << it2->parent.attribute << ") == \"" << it2->label << "\") && yetToPrint && (child == " << it2->parent.attribute << ")) { \n";
        outfile << "child = " << it2->child.attribute <<"; \n";
        outfile << "if (" << it2->child.attribute << " == -1) { \n";
                    
        outfile << "PrintExampleRow(*it1, \"" << it2->child.label << "\"); \n";
        outfile << "yetToPrint = false; \n";
                    
        outfile << "} \n";
    
        outfile << "} \n";
        oldAttribute = it2->parent.attribute;
    }
       
    arcs = arcsCopy;
    outfile << "yetToPrint = true; \n";
    outfile << "firstPass = true; \n";
    outfile << "} \n";
    outfile << "} \n";
    
    outfile << "void PrintExampleRow (vector< string> & exampleRow, string label) { \n";
    outfile << "for (vector<string>::iterator it1 = exampleRow.begin(); it1 != exampleRow.end(); ++it1) { \n";
    outfile << "cout << *it1 << \" \"; \n";
    outfile << "} \n";
    outfile << "cout << \" ==> \" << label << endl; \n";
    outfile << "} \n";
    
    outfile << "void StringToVector (string str, vector<string> & vec) { \n";
    outfile << "int pos = -1; \n";
    outfile << "do { \n";
    outfile << "pos = (int)str.find(','); \n";
    outfile << "string token = str.substr(0,pos); \n";
    outfile << "token = ClearSpacesBefore(token); \n";
    outfile << "token = ClearSpacesAfter(token); \n";
    outfile << "vec.push_back(token); \n";
    outfile << "str = str.substr(pos+1); \n";
    outfile << "} while (pos >= 0); \n";
    outfile << "} \n";
    
    outfile << "string ClearSpacesBefore (string str) { \n";
        outfile << "while (iswspace(str[0])) str = str.substr(1); \n";
        outfile << "return str; \n";
    outfile << "} \n";
    
    outfile << "string ClearSpacesAfter (string str) { \n";
        outfile << "while (iswspace(str[str.length()-1])) str = str.substr(0,str.length()-1); \n";
        outfile << "return str; \n";
    outfile << "} \n";
    
    outfile.close();
}

// The functions below do the same job as output .cpp file nfi2103-decisionTree.cpp, that is,
// prints the classification labels of examples in a test data set.

/*
void AssignLabels (vector< decisionTreeArc> & arcs, vector< vector<string> > & examples) {
    vector< decisionTreeArc> arcsCopy = arcs;
    outfile << "void AssignLabels (vector< vector<string> > & examples) { \n";
    //cout << \"ASSIGN LABELS\" << endl; \n";
    //cout << \"Print Arcs:\" << endl; \n";
    //PrintArcs(arcs);
    //cout << \"Print Examples:\" << endl; \n";
    //PrintExamples(examples); \n";
    //cout << \"RESULTS\" << endl; \n";
    for (vector< vector<string> >::iterator it1 = examples.begin(); it1 != examples.end(); ++it1) {
        outfile << "for (vector< vector<string> >::iterator it1 = examples.begin(); it1 != examples.end(); ++it1) { \n";
        //if (examples.size() == 0) break; \n";
        //PrintExampleRow(*it1,""); \n";
        for (vector <decisionTreeArc>::iterator it2 = arcs.begin(); it2 != arcs.end(); ++it2) { 
            //cout << \"it2->parent.attribute is \" << it2->parent.attribute << endl; \n";
            //cout << \"it1->at(it2->parent.attribute) is \" << it1->at(it2->parent.attribute) << endl; \n";
            //cout << \"it2->label is \" << it2->label << endl; \n";
            if (it1->at(it2->parent.attribute) == it2->label) {
                outfile << "if (it1->at(" << it2->parent.attribute << ") == \"" << it2->label << "\") {" \n";
                if (it2->child.attribute == -1) {
                    outfile << "if (" << it2->child.attribute << " == -1) { \n"; 
                    //PrintExampleRow(*it1, it2->child.label);
                    outfile << "PrintExampleRow(*it1, \"" << it2->child.label << "\"); \n";
                    break;
                    outfile << "} \n";
                } 
                else if (!arcs.empty()) { 
                    int i = 0; 
                    while(arcs.at(i).parent.attribute == it2->parent.attribute) { 
                        arcs.erase(arcs.begin()+i); 
                        i++; 
                    }
                }
                outfile << "} \n";
            } 
        } 
        arcs = arcsCopy;
        outfile << "} \n";
    }
    outfile << "} \n";
} 

void PrintExampleRow (vector< string> & exampleRow, string label) {
    for (vector<string>::iterator it1 = exampleRow.begin(); it1 != exampleRow.end(); ++it1) {
        cout << *it1 << " ";
    }
    cout << " ==> " << label << endl;
}


*/












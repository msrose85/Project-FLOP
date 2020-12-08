import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import filedialog, WORD
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pickle
from pickle import dump
from pickle import load
from sklearn.ensemble import RandomForestClassifier
from tkinter import ttk
import os

class VPN_predictor:
    
    def open_file():
        """Open a file for training."""
        global dataset, X_train, X_test, y_train, y_test, sc_X, Y_axis
        filepath = askopenfilename(
            filetypes=[("csv Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        txt_edit.delete("1.0", tk.END)
        with open(filepath, "r") as input_file:
            dataset = pd.read_csv(input_file)
        
        try:
            # pre-processing    
            dataset = dataset[dataset.Duration != 0]
            X = dataset.iloc[:,np.r_[8,27,32,31,36,30,35,28,33,22,21,26,25,23,24,79,76,78,77,83,80,82,81,83]].values # change to all features if running the preliminary model
            y = dataset.iloc[:, 84].values
            from sklearn.preprocessing import LabelEncoder
            labelencoderY = LabelEncoder()
            y = labelencoderY.fit_transform(y)
            Y_axis = np.unique(labelencoderY.inverse_transform(y))
            #save Y_Axis
            np.save("Y_axis", Y_axis)
        
            # Splitting the dataset into Training and Test sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        
            # Feature Scaling 
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            #save the scaling factors
            dump(sc_X, open('scaler.pkl', 'wb'))
        
            # print confirmation
            text = (f"Training Data Loaded - {filepath}")
            txt_edit.insert(tk.END, text)
        except:
            text = ("An error occured - please ensure you are loading a correctly formatted csv")
            txt_edit.insert(tk.END, text)   
            
    def train_model():
        global model
        try:
            rf = RandomForestClassifier()
            # Refine the performance of the model by adding details here - adding more will increase training time
            grid = {'n_estimators': [ 800]}    
            # Cross Validation and Training
            rf_grid= RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter=30, cv = 5, verbose=2, n_jobs = -1)
            model = rf_grid.fit(X_train, y_train)
            # save the model to disk
            filename = 'RF VPN Predictor.sav'
            pickle.dump(model, open(filename, 'wb'))
           
            # print confirmation
            text = ("\n\n model has been trained and saved")
            txt_edit.insert(tk.END, text)
        except:
            text = ("\n\n sorry - that didn't work', have you loaded training data yet?")
            txt_edit.insert(tk.END, text)
            
    def open_model():
        """Open an existing model"""
        global model, sc_X, Y_axis
        filepath = askopenfilename(
            filetypes=[("models", "*.sav"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, "rb") as input_file:
                model = pickle.load(input_file)
                # print confirmation
            text = ("\n\n Model has been loaded")
            txt_edit.insert(tk.END, text)
            sc_X = load(open('scaler.pkl', 'rb'))
            Y_axis = np.load("Y_axis.npy", allow_pickle=True)
        except:
            text = ("\n\n sorry - that didn't work', please try again")
            txt_edit.insert(tk.END, text) 
        
    def open_test():
        """Open a file for training."""
        global X_new, testset
        filepath = askopenfilename(
            filetypes=[("csv Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        with open(filepath, "r") as input_file:
            testset = pd.read_csv(input_file)
        
        try:    
            #pre-processing    
            # splitting variables in the new testset
            X_new = testset.iloc[:,np.r_[7,26,31,30,35,29,34,27,32,21,20,25,24,22,23,78,75,77,76,82,79,81,80,82]].values
        
            # Feature scaling
            X_new = sc_X.transform(X_new)
            
            text = ("\n\n Test data loaded and ready to make predictions")
            txt_edit.insert(tk.END, text)
        except:
            text = ("\n\n An error occured - please ensure you are loading a correctly formatted csv")
            txt_edit.insert(tk.END, text)     
    
        
    def make_pred():
        try:
            y_pred_RF_new = model.predict(X_new)
        
            # Replace the labels
            index = []
            for i in range(len(Y_axis)):
                index.append(i)
            key_index = np.array(index)
            key_index = np.column_stack((key_index, Y_axis))
            key_index = pd.DataFrame(key_index)
            #pred = pd.DataFrame(y_pred_RF_new)
            results = []
            for r in y_pred_RF_new:
                for i in key_index[0]:
                    if r == i:
                        results.append(key_index.iloc[i,1])
        
            #Add the predictions to the testset
            testset['predictions'] = results
            simple_results = testset.iloc[0:5,np.r_[1,2,3,4,84]]
           
        
            text = (f"\n\n Showing first few lines of the predictions \n\n {simple_results}")
            txt_edit.insert(tk.END, text)
        except:
            text = ("\n\n Sorry - that didn't work  - please try again")
            txt_edit.insert(tk.END, text)   
      
    def save_file():
        """Save the current output as a file."""
    
        export_file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("csv", "*.csv"), ("All Files", "*.*")])
        testset.to_csv (export_file_path, index = False, header=True)
        
    def clear_panel():
        txt_edit.delete('1.0', tk.END)
        
def start_cic():
    os.startfile('CICFlowMeter-4.0\\bin\\CICFlowMeter.bat')

# intializing the Window
window = tk.Tk()
window.title("VPN Predictor")


#Create Tab Control
TAB_CONTROL = ttk.Notebook(window)
#Tab1
TAB1 = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(TAB1, text='Guide')
#Tab2
TAB2 = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(TAB2, text='VPN Predictor')
TAB_CONTROL.pack(expand=1, fill="both")
#Tab3
TAB3 = ttk.Frame(TAB_CONTROL)
TAB_CONTROL.add(TAB3, text='Feature Importance')
TAB_CONTROL.pack(expand=1, fill="both")

#Tab 1

f = open("guide.txt", "r")
guide_text = f.read()
f.close()

Scroll = tk.Scrollbar(TAB1)
Scroll.pack(side="right", fill="y")
guide = tk.Text(master=TAB1, font='Helvetica 12')
guide.config(wrap=WORD)
guide.pack(expand = 1, fill="both")

Scroll.config(command=guide.yview)
guide.config(yscrollcommand=Scroll.set)
guide.insert(tk.END, guide_text)

guide.tag_add('first','1.0','1.end')
guide.tag_config('first', font='Helvetica 12 bold')

guide.tag_add('notes','11.0','11.end')
guide.tag_config('notes', font='Helvetica 12 bold')

guide.tag_add('features','21.0','21.end')
guide.tag_config('features', font='Helvetica 12 bold')

btn_cic = tk.Button(
    TAB1,
    text="  Open Cic-Flow-Meter  ",
    bg="#008ddd",
    fg="yellow",
    command=start_cic,
)

btn_cic.pack(side="left") 

#Tab 2

TAB2.rowconfigure((0,1), minsize=170, weight=2)
TAB2.rowconfigure(2, weight = 1)
#TAB2.columnconfigure(1, weight=1)
TAB2.columnconfigure((0,1,2), minsize=250, weight=1)

# Command Buttons
fr_buttons_left = tk.Frame(TAB2)
fr_buttons_right = tk.Frame(TAB2)
fr_buttons_middle = tk.Frame(TAB2)

label_start = tk.Label(master=fr_buttons_left, text="\nHaven't trained a model yet? \nStart here",font='Helvetica 14 bold')

btn_load_train = tk.Button(
    fr_buttons_left,
    text="  Select training data to train a new model  ",
    bg="#008ddd",
    fg="yellow",
    command=VPN_predictor.open_file,
                    )
btn_train = tk.Button(
    fr_buttons_left,
    text="  Train the model  ",
    bg="#008ddd",
    fg="yellow",
    command = VPN_predictor.train_model,
                    )
label_a = tk.Label(master=TAB2, text="\n\nOr", font='Helvetica 14 bold')

label_load = tk.Label(master=fr_buttons_right, text="Already have a model trained?\n", font='Helvetica 14 bold')

btn_load_model = tk.Button(
    fr_buttons_right,
    text="  Load an existing model  ",
    bg="#008ddd",
    fg="yellow",
    command=VPN_predictor.open_model,
                    )
label_b = tk.Label(master=fr_buttons_middle, text="then", font='Helvetica 14 bold')

btn_load_test = tk.Button(
    fr_buttons_middle,
    text="  Select test data  ",
    bg="#008ddd",
    fg="yellow",
    command=VPN_predictor.open_test,
                    )

btn_test = tk.Button(
    fr_buttons_middle,
    text="  Make predictions!  ",
    bg="#008ddd",
    fg="yellow",
    command = VPN_predictor.make_pred
                    )

btn_save = tk.Button(
    fr_buttons_middle,
    text="  Save results  ",
    bg="#008ddd",
    fg="yellow",
    command = VPN_predictor.save_file,
                    )

# Display area
txt_edit = tk.Text(TAB2)

btn_clear = tk.Button(
    TAB2,
    text="  Clear panel  ",
    bg="gray",
    fg="yellow",
    command = VPN_predictor.clear_panel,
)


#Place objects
label_start.grid(row=0, column=0, sticky="", padx=5, pady=5)

btn_load_train.grid(row=1, column=0, sticky="", padx=5, pady=5)

btn_train.grid(row=2, column=0, sticky="", padx=5, pady=5)

label_a.grid(row=0, column=1, sticky="n", padx=5, pady=5)

label_load.grid(row=0, column=0, sticky="n", padx=5, pady=5)

btn_load_model.grid(row=1, column=0, sticky="", padx=5, pady=5)

label_b.grid(row=0, column=0, sticky="n", padx=5, pady=(20,5))

btn_load_test.grid(row=1, column=0, sticky="", padx=5, pady=5)

btn_test.grid(row=2, column=0, sticky="", padx=5, pady=5)

btn_save.grid(row=3, column=0, sticky="", padx=5, pady=(5, 30))

fr_buttons_left.grid(row=0, column=0, sticky="")
fr_buttons_right.grid(row=0, column=2, sticky="")
fr_buttons_middle.grid(row=1, column=1, sticky="")

txt_edit.grid(row=2, column=0, columnspan =3, sticky="nsew")
btn_clear.grid(row=2, column=0, columnspan =3, sticky="se")

#TAB2.mainloop()

#TAB3

TAB3.rowconfigure((0,1), minsize=170, weight=2)
TAB3.rowconfigure(2, weight = 1)
#TAB2.columnconfigure(1, weight=1)
TAB3.columnconfigure((0,1,2), minsize=250, weight=1)

# Command Buttons
fr_buttons_left_3 = tk.Frame(TAB3)
fr_buttons_right_3 = tk.Frame(TAB3)
fr_buttons_middle_3 = tk.Frame(TAB3)

def open_feature():
        """Open a file for training."""
        global feature_set, X_train, X_test, y_train, y_test, sc_X, Y_axis
        filepath = askopenfilename(
            filetypes=[("csv Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return
        txt_edit.delete("1.0", tk.END)
        with open(filepath, "r") as input_file:
            feature_set = pd.read_csv(input_file ,low_memory=False)
        
        try:
            # pre-processing    
            feature_set = feature_set[feature_set.Duration != 0]
            X = feature_set.iloc[:,np.r_[8,27,32,31,36,30,35,28,33,22,21,26,25,23,24,79,76,78,77,83,80,82,81,83]].values # change to all features if running the preliminary model
            y = feature_set.iloc[:, 84].values
            from sklearn.preprocessing import LabelEncoder
            labelencoderY = LabelEncoder()
            y = labelencoderY.fit_transform(y)
            Y_axis = np.unique(labelencoderY.inverse_transform(y))
            #save Y_Axis
            np.save("Y_axis", Y_axis)
        
            # Splitting the feature_set into Training and Test sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        
            # Feature Scaling 
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            #save the scaling factors
            dump(sc_X, open('scaler.pkl', 'wb'))
        
            # print confirmation
            text = (f"Training Data Loaded - {filepath}")
            txt_edit_feature.insert(tk.END, text)
        except:
            text = ("\n\nAn error occured - please ensure you are loading a correctly formatted csv")
            txt_edit_feature.insert(tk.END, text)  
            
  
def Feature_Ranking():
    global fig
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        import matplotlib.pyplot as plt
        
        forest = ExtraTreesClassifier(n_estimators=800,
                                      random_state=0)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        
        #label indicies for the plot
        feature_set_headers = feature_set.iloc[0:0,np.r_[8,27,32,31,36,30,35,28,33,22,21,26,25,23,24,79,76,78,77,83,80,82,81,83]]
        headers = list(feature_set_headers.columns)
        h=[]
        for i in indices:
            h.append(headers[i])
            
        # Print the feature ranking
        #print("Feature ranking:")
        #for f in range(X_train.shape[1]):
        #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        
        # Plot the impurity-based feature importances of the forest
        fig = plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color=(0.2, 0.4, 0.6, 0.6), yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), h, rotation='vertical')
        plt.xlim([-1, X_train.shape[1]])
               
        # print confirmation
        text = ("\n\nRanking Comple")
        txt_edit_feature.insert(tk.END, text)
    except:
        text = ("\n\nAn error occured - please ensure you are loading a correctly formatted csv")
        txt_edit_feature.insert(tk.END, text)  

def calculating():
    text = ("\n\nCalculating - this could take a few minutes")
    txt_edit_feature.insert(tk.END, text)
    window.after(100,Feature_Ranking)  
    
def show_Plot():
    fig.show()
    
label_features_desc = tk.Label(master=fr_buttons_left_3, text="Show the relative importance of each feature in a dataset", font='Helvetica 14 bold')

    
btn_load_feature = tk.Button(
    fr_buttons_left_3,
    text="  Select dataset  ",
    bg="#008ddd",
    fg="yellow",
    command=open_feature,
                    )
    
btn_feature = tk.Button(
    fr_buttons_left_3,
    text="  Calculate Feature Importance  ",
    bg="#008ddd",
    fg="yellow",
    command = calculating
    )


btn_show_features = tk.Button(
    fr_buttons_left_3,
    text="  Show Plot  ",
    bg="#008ddd",
    fg="yellow",
    command = show_Plot,
    )

# Display area
txt_edit_feature = tk.Text(TAB3)

btn_clear_feature= tk.Button(
    TAB3,
    text="  Clear panel  ",
    bg="gray",
    fg="yellow",
                    )
# Display area
txt_edit = tk.Text(TAB2)

btn_clear = tk.Button(
    TAB2,
    text="  Clear panel  ",
    bg="gray",
    fg="yellow",
    command = VPN_predictor.clear_panel,
)


label_features_desc.grid(row=1, column=0, sticky="n", padx=5, pady=(20,5))
btn_load_feature.grid(row=2, column=0, sticky="W", padx=5, pady=(5, 30))
btn_feature.grid(row=3, column=0, sticky="W", padx=5, pady=(5, 30))
btn_show_features.grid(row=4, column=0, sticky="W", padx=5, pady=(5, 30))



fr_buttons_left_3.grid(row=0, column=0, sticky="")
fr_buttons_right_3.grid(row=0, column=2, sticky="")
fr_buttons_middle_3.grid(row=1, column=1, sticky="")

txt_edit_feature.grid(row=2, column=0, columnspan =3, sticky="nsew")

window.mainloop()


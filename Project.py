#%%
# ------------------------------------------------------------------------------ #
                            # Unsupervised Learning 2019
                            # Author: Yuval Lavie
# ------------------------------------------------------------------------------ #
#%%
# ------------------------------------------------------------------------------ #
                            # Imports
# ------------------------------------------------------------------------------ #
import numpy as np
import pandas as pd;
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt;
import seaborn as sns
from sklearn import preprocessing as pp
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import shuffle


# Set the seed for reproducability
rand_state = 2020;
np.random.seed(rand_state);
# Load the data
data = pd.read_csv("creditcard.csv");
# Separate the features from the labels
features = data.iloc[:,1:30].copy()
labels = data.iloc[:,30].copy()

def Visualize(data,labels,algorithm):
    plt.figure(figsize=(10,5));
    plt.grid();
    plt.title("First two dimensions of " + str(algorithm));
    plt.scatter(data[:,0],data[:,1],c=labels)
    plt.show()

#%%
# ------------------------------------------------------------------------------ #
                            # Descriptive Statistics
# ------------------------------------------------------------------------------ #

# Shallow statistics
stats = data.describe();
print(stats)

# Missing values ?
print("Are there any missing values? answer:",data.isnull().values.any())

# Just how imbalanced is the data ?
plt.figure();
plt.hist(labels);
plt.xlabel("Class");
plt.ylabel("Frequency");
plt.title("Distribution of classes")
plt.grid();
plt.show()

# Ratio of frauds
frauds = labels[labels == 1];
print("Frauds ratio:",len(frauds) / len(labels))

# Can we visually see the relationship between the amount of money and the label?
plt.scatter(data.index.values,data['Amount'],c=labels)
plt.show()
print("No we cant");


# How do the features distribution look like?
features.hist(figsize=(20,20))
plt.show()

# Correlation Matrix

# Generate a large random dataset
d = features

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True,fmt=".2f",
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


corr = features.corrwith(data['Class']).reset_index()
corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = False)
plt.figure(figsize=(4,15))
fig = sns.heatmap(corr, annot=True, fmt="g", cmap='YlGnBu')
plt.title("Correlation of Variables with Class")
plt.show()

#%%
# Let's see the relation of V17 and V14 to the label
plt.figure(figsize=(10,5));
plt.grid();
plt.xlabel("V17");
plt.ylabel("V14");
plt.scatter(features['V17'],features['V14'],c=labels)
plt.show()

#%%
print("Seems like the data can be separated almost only by these two labels.")
print("Should we build a classifier based on these two or the 5-6 highest, but then is that a supervised method?")

num_of_features = 6;
correlated_features = np.abs(corr).sort_values(by=['Correlations']);
descriptive_features = correlated_features.iloc[len(correlated_features) - num_of_features:len(correlated_features)]
new_features = data[descriptive_features.index.values];
#%%
# ------------------------------------------------------------------------------ #
                        # Additional Visualization Models
# ------------------------------------------------------------------------------ #
#%% Principal Component Analysis
# Since the data is already a PCA of an original data set we visualize the first two vectors.
Visualize(features.to_numpy(),labels,"PCA")
#%% Sparse Random Projection
from sklearn.random_projection import SparseRandomProjection
SRP = SparseRandomProjection(n_components=2,dense_output = True,random_state=rand_state);
SRP_features = SRP.fit_transform(features);
Visualize(SRP_features,labels,"SRP")

#%% Gaussian Random Projection
from sklearn.random_projection import GaussianRandomProjection
GRP = GaussianRandomProjection(n_components=2,random_state=rand_state);
GRP_features = GRP.fit_transform(features);
Visualize(GRP_features,labels,"GRP")

#%% Auto Encoder
import torch;
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X_tensor = torch.from_numpy(features.to_numpy()).type(torch.FloatTensor)

loader = torch.utils.data.DataLoader(
        X_tensor, batch_size=128, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None)

class Net(nn.Module):
    def __init__(self):
        # super Constructor
        super(Net, self).__init__()

        # Define the networks architecture
        self.encoder = nn.Sequential(
            nn.Linear(29, 27),
            nn.ReLU(inplace=True),
            nn.Linear(27, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25,20)
            );

        self.decoder = nn.Sequential(
            nn.Linear(20, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 27),
            nn.ReLU(inplace=True),
            nn.Linear(27,29)
            );


    def setCriterion(self,criterion):
        self.criterion = criterion;

    def setOptimizer(self,optimizer):
        self.optimizer = optimizer;


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x;

    def transform(self,x):
        return self.encoder(x);

    def inverse_transform(self,x):
        return self.decoder(x);

    def fit(self,X_train,epochs):
        for epoch in range(epochs):
            for batch_idx, x in enumerate(loader):
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward propogation
                output = self.forward(x)

                # Calculate the loss
                loss = self.criterion(output, x)

                # Calculate the derivatives
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Print the batch loss
                if(batch_idx % 2000 == 0):
                    print(f"Epoch: {epoch} Loss: {loss.item()}")

        print("Training completed")

# Initialize the network
clf = Net()

# Print the network
print(clf)

# Set the loss criterion
criterion = F.mse_loss;
clf.setCriterion(criterion);

# Set the optimizer
optimizer = optim.Adam(params=clf.parameters(), lr=0.001);
clf.setOptimizer(optimizer);

# Set the network to training mode
clf.train();
# Train the network
clf.fit(loader,20);

# Encode the data and visualize it
reduced_features = clf.transform(X_tensor).detach().numpy()

Visualize(reduced_features,labels,"Autoencoder")

#%%
# ------------------------------------------------------------------------------ #
                                # Data Preprocessing
# ------------------------------------------------------------------------------ #
scaler = pp.StandardScaler(copy=True);
anomaly_features = pd.DataFrame(scaler.fit_transform(new_features))


# Split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(anomaly_features, labels, test_size=0.2, \
                    random_state=rand_state, stratify=labels)
#%%
# ------------------------------------------------------------------------------ #
                            # Anomaly Detection
# ------------------------------------------------------------------------------ #

def plotResults(trueLabels, anomalyScores, returnPreds = False):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = \
        precision_recall_curve(preds['trueLabel'],preds['anomalyScore'])
    average_precision = \
        average_precision_score(preds['trueLabel'],preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = \
    {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], \
                                     preds['anomalyScore'])
    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: \
    Area under the curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.show()

    if returnPreds==True:
        return preds

def anomaly_scores(features,reconstructed_features):
    loss = np.sum((np.array(features)-np.array(reconstructed_features))**2, axis=1)
    return pd.Series(data=loss,index=features.index);


# Principal Component Analysis
from sklearn.decomposition import PCA

# Initialize the transformer
pca = PCA(n_components=5,random_state=rand_state)

# Fit the transformer on the training set
pca.fit(X_train);

# Transform the training set
transformed_training_set = pca.transform(X_train);

# Reconstructed training set
reconstructed_training_set = pca.inverse_transform(transformed_training_set);

# Anomaly scores for each sample based on the original data and the reconstructed set
train_anomaly_scores = anomaly_scores(X_train,reconstructed_training_set);


precision, recall, thresholds = \
    precision_recall_curve(y_train,train_anomaly_scores)

# Detect anomalies by finding the best treshold and applying it.
preds = plotResults(y_train, train_anomaly_scores, True)


alpha = thresholds[np.where(precision > 0.8)[0][0]]

y_train_pred = np.where(train_anomaly_scores > alpha,1,0)


target_names = ['Legitimate', 'Fraud']
print(classification_report(y_train, y_train_pred, target_names=target_names))

ax1 = plt.subplot()
sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True,fmt='g', ax = ax1);

# labels, title and ticks
ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels');
ax1.set_title('Confusion Matrix');
ax1.xaxis.set_ticklabels(target_names)
ax1.yaxis.set_ticklabels(target_names);

plt.show()


# Transform the test set
transformed_test_set = pca.transform(X_test);

# Reconstructed training set
reconstructed_test_set = pca.inverse_transform(transformed_test_set);

# Anomaly scores for each sample based on the original data and the reconstructed set
test_anomaly_scores = anomaly_scores(X_test,reconstructed_test_set);

y_test_pred = np.where(test_anomaly_scores > alpha,1,0)


print("Evaluation Metrics")
print("-----------------------------------")
# Evaluation Indices
evaluators = {};

# Adjusted Mutual Information
evaluators['mutual_info'] = metrics.adjusted_mutual_info_score(y_test, y_test_pred)
print("Adjusted Mutual Information: %0.3f"
      % evaluators['mutual_info'])

# Adjusted Mutual Information
evaluators['jaccard'] = metrics.jaccard_score(y_test,y_test_pred)
print("Jaccard Score: %0.3f"
      % evaluators['jaccard'])

# Statistical Significance

# For example lets check if 0.593 V-measure is good or not by creating a distribution of V-measures.

# Number of shuffles to create the distribution
M = 10**3;
# Name of our statistics
statistics_names = ['mutual_info','jaccard'];
# Skeleton for our statistics
values = np.zeros((len(statistics_names),M))

# Create the distribution for each statistic
for i in range(M):
    shuffled_labels = shuffle(y_test);
    evals = np.array([metrics.adjusted_mutual_info_score(shuffled_labels, y_test_pred),metrics.jaccard_score(shuffled_labels,y_test_pred) ])
    values[:,i] = evals;

# Check whether any statistics may be random.
for i in range(len(statistics_names)):
    print(f"{statistics_names[i]}: Maximum Value: {np.max(values[i])} Our value: {evaluators[statistics_names[i]]}")

print("-----------------------------------")

target_names = ['Legitimate', 'Fraud']
print(classification_report(y_test, y_test_pred, target_names=target_names))

ax1 = plt.subplot()
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True,fmt='g', ax = ax1);

# labels, title and ticks
ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels');
ax1.set_title('Confusion Matrix');
ax1.xaxis.set_ticklabels(target_names)
ax1.yaxis.set_ticklabels(target_names);

plt.show()


x = X_test.index.values;
plt.figure(figsize=(10,5));
plt.grid();
plt.xlabel("Transaction Index")
plt.ylabel("Anomaly Score")
plt.title("Transaction score with the threshold")
plt.plot(x,np.ones(len(x))*alpha,label="Threshold")
plt.scatter(x,test_anomaly_scores,c=y_test)
plt.legend()
#%%
# ------------------------------------------------------------------------------ #
                                # Clustering
# ------------------------------------------------------------------------------ #
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# Data Preprocessing
clustering_features = features[['V17','V14']];
scaler = pp.StandardScaler(copy=True);
clustering_features = pd.DataFrame(scaler.fit_transform(clustering_features.to_numpy()))
Visualize(clustering_features.to_numpy(),labels,"Original")

# Split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(clustering_features.to_numpy(), labels, test_size=0.2, \
                    random_state=rand_state, stratify=labels)
#%%
X = X_train;
y = y_train;
X_test = np.array(X_test);
y_test = np.array(y_test);

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    print(f"K-Means for {n_clusters} started");
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=rand_state)
    print("Fitting and predicting..")
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    print("Calculating silhouettes")
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    print("Calculating samples silhouette")
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    print("Plotting..")
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()
#%%
# Let's verify this on the test set.

num_of_clusters = 3;
print(f"Evaluating our clusters with {num_of_clusters} centers")
clusterer = KMeans(n_clusters=num_of_clusters, random_state=rand_state)
cluster_labels = clusterer.fit_predict(X);

test_labels = clusterer.predict(X_test);

# Get the minimal group indices
index_of_anomalies = np.argmin(np.unique(test_labels,return_counts = True)[1]);

#Visualize(X_test,test_labels,"K-Means " + str(num_of_clusters) + " Clusters")
#Visualize(SRP_features,labels.to_numpy(),"Original")

# It makes sense that there are many type of legitimate transactions which are alike and a different type which is an anomaly.
y_pred = np.where(test_labels == index_of_anomalies,1,0)

evaluators = {};
print("Calculating evaluators")
# Silhouette Score
evaluators['Silhouette'] = metrics.silhouette_score(X_test, y_pred)
print("Silhouette Coefficient: %0.3f"
      % evaluators['Silhouette'] );
# Homogeneity Metric
evaluators['homogeneity'] = metrics.homogeneity_score(y_test, y_pred)
print("Homogeneity: %0.3f" % evaluators['homogeneity'])

# Completeness Metric
evaluators['completeness'] = metrics.completeness_score(y_test, y_pred)
print("Completeness: %0.3f" % evaluators['completeness'])

# V Score (Mutual Information)
evaluators['v_measure'] = metrics.v_measure_score(y_test, y_pred)
print("V-measure: %0.3f" % evaluators['v_measure'] )

# Adjusted Rand Index
evaluators['rand_index'] = metrics.adjusted_rand_score(y_test, y_pred)
print("Adjusted Rand Index: %0.3f"
      % evaluators['rand_index'])

# Adjusted Mutual Information
evaluators['mutual_info'] = metrics.adjusted_mutual_info_score(y_test, y_pred)
print("Adjusted Mutual Information: %0.3f"
      % evaluators['mutual_info'])

# Adjusted Mutual Information
evaluators['jaccard'] = metrics.jaccard_score(y_test,y_pred)
print("Jaccard Score: %0.3f"
      % evaluators['jaccard'])
print("---------------------------")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


target_names = ['Legitimate', 'Fraud']
print(classification_report(y_test, y_pred, target_names=target_names))

ax1 = plt.subplot()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,fmt='g', ax = ax1);

# labels, title and ticks
ax1.set_xlabel('Predicted labels');
ax1.set_ylabel('True labels');
ax1.set_title('Confusion Matrix');
ax1.xaxis.set_ticklabels(target_names)
ax1.yaxis.set_ticklabels(target_names);

plt.show()
#%% Statistical Significance
from sklearn.utils import shuffle
# For example lets check if 0.593 V-measure is good or not by creating a distribution of V-measures.

# Number of shuffles to create the distribution
M = 10**3;
# Name of our statistics
statistics_names = ['homogeneity','completeness','v_measure','rand_index','mutual_info','jaccard'];
# Skeleton for our statistics
values = np.zeros((len(statistics_names),M))

# Create the distribution for each statistic
for i in range(M):
    shuffled_labels = shuffle(y_test);
    evals = np.array([metrics.homogeneity_score(shuffled_labels, y_pred),metrics.completeness_score(shuffled_labels, y_pred),metrics.v_measure_score(shuffled_labels, y_pred),metrics.adjusted_rand_score(shuffled_labels, y_pred),metrics.adjusted_mutual_info_score(shuffled_labels, y_pred),metrics.jaccard_score(shuffled_labels,y_pred) ])
    values[:,i] = evals;

# Check whether any statistics may be random.
for i in range(len(statistics_names)):
    print(f"{statistics_names[i]}: Maximum Value: {np.max(values[i])} Our value: {evaluators[statistics_names[i]]}")
#%%
# from sklearn.svm import OneClassSVM

# # Prepare the data for OneClassSVM
# # The non-fraudulent cases are the majority and we learn them only. (This is semi-supervised?)

# # Train the model
# clf = OneClassSVM(gamma='auto').fit(X_train)
# y_pred = clf.predict(X_train)
# y_pred = np.where(y_pred == 1,1,0)
# plot_results(y_train,y_pred,"One Class SVM");

# #%% Isolation Forest
# from sklearn.ensemble import IsolationForest
# clf = IsolationForest(random_state=rand_state).fit(X_train)
# y_pred = clf.predict(X_train)
# y_pred = np.where(y_pred == 1,1,0)
# plot_results(y_train,y_pred,"Isolation Forest");
# #%% Local Outlier Factor
# from sklearn.neighbors import LocalOutlierFactor
# clf = LocalOutlierFactor(n_neighbors=2)
# y_pred = clf.fit_predict(features)
# y_pred = np.where(y_pred == 1,1,0)
# plot_results(y_train,y_pred,"Local Outlier Factor");


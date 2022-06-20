#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

typedef struct NODE {
	char item;
	char arrnode[200];
	int num;
	struct NODE* left;
	struct NODE* right;
} node;

typedef struct LIST {
	node* thisnode;
	struct LIST* next;
} list;

list* FullList = NULL;

list* Insert(node* newnode) {
	list* new = NULL;
	new = (list*)malloc(sizeof(list));
	new->thisnode = newnode;
	new->next = NULL;

	if (FullList == NULL) {
		FullList = new;
		return FullList;
	}
	else if (newnode->num < FullList->thisnode->num) {
		new->next = FullList;
		FullList = new;
		return FullList;
	}
	else {
		list* temp = FullList;
		while (temp->next != NULL) {
			if (newnode->num < temp->next->thisnode->num) {
				new->next = temp->next;
				temp->next = new;
				break;
			}
			else
				temp = temp->next;
		}
		if (temp->next == NULL)
			temp->next = new;
		return temp;
	}
	return 0;
}

void Finder(node* node, char item, char* result, int num, FILE* fp) {
	if (node) {
		num++;
		result[num] = '0';
		Finder(node->left, item, result, num, fp);
		result[num] = '1';
		Finder(node->right, item, result, num, fp);
		result[num] = '\0';
		if (node->item == item) {
			printf("%s", result);
			fprintf(fp, "%s", result);
		}
	}
	return 0;
}

int main() {
	char input;
	char arr0[1000000];
	char arr1[26];
	int arr2[26] = {0, };
	int counter, arr_count = 0, arr0_count = 0;

	FILE* txt = fopen("hw3_input.txt", "r");
	while (1) {
		if ((fscanf(txt, "%c", &input)) == -1) {
			break;
		}
		if (isalpha(input) == 1 || isalpha(input) == 2) {
			input = tolower(input);
			arr0[arr0_count] = input;
			arr0_count++;
			counter = 0;
			for (int i = 0; i < 26; i++) {
				if (input == arr1[i]) {
					arr2[i]++;
					counter = 1;
					break;
				}
			}
			if (counter == 0) {
				arr1[arr_count] = input;
				arr2[arr_count]++;
				arr_count++;
			}
		}
	}
	fclose(txt);

	int arr_arrange = arr_count;
	for (int i = 0; i < 26; i++) {
		if (arr2[i] == 0) {
			counter = i - 1;
			break;
		}
		node* newnode = (node*)malloc(sizeof(node));
		newnode->item = arr1[i];
		newnode->num = arr2[i];
		newnode->left = NULL;
		newnode->right = NULL;
		newnode->arrnode[0] = arr1[i];
		for (int j = 1; j < 200; j++)
			newnode->arrnode[j] = '#';
		Insert(newnode);
	}

	for (int i = 0; i < counter; i++) {
		node* node1 = FullList->thisnode;
		FullList = FullList->next;
		node* node2 = FullList->thisnode;
		FullList = FullList->next;
		node* sum = (node*)malloc(sizeof(node));
		sum->item = '+';
		sum->num = node1->num + node2->num;
		sum->left = node1;
		sum->right = node2;

		for (int j = 0; j < 200; j++)
			sum->arrnode[j] = '#';
		sum->arrnode[0] = '(';
		arr_count = 0;
		int arr_num = 1;
		for (int j = 0; j < 200; j++) {
			if (node1->arrnode[j] == '#')
				break;
			arr_count++;
		}
		for (int j = 0; j < arr_count; j++) {
			sum->arrnode[arr_num] = node1->arrnode[j];
			arr_num++;
		}
		sum->arrnode[arr_num] = ',';
		arr_num++;
		arr_count = 0;
		for (int j = 0; j < 200; j++) {
			if (node2->arrnode[j] == '#')
				break;
			arr_count++;
		}
		for (int j = 0; j < arr_count; j++) {
			sum->arrnode[arr_num] = node2->arrnode[j];
			arr_num++;
		}
		sum->arrnode[arr_num] = ')';

		Insert(sum);
	}

	node* final = (node*)malloc(sizeof(node));
	final = FullList->thisnode;
	for (int i = 0; i < 200; i++) {
		if (final->arrnode[i] == '#')
			break;
		printf("%c", final->arrnode[i]);
	}

	FILE* fp = fopen("hw3_output1.txt", "w");
	for (int i = 0; i < 200; i++) {
		if (final->arrnode[i] == '#')
			break;
		fprintf(fp, "%c", final->arrnode[i]);
	}
	fprintf(fp, "%s", "HEADEREND");
	printf("HEADEREND");
	char* result = (char*)malloc(sizeof(char));
	for (int i = 0; i < arr0_count; i++) {
		Finder(final, arr0[i], result, -1, fp);
	}
	fclose(fp);

	printf("\n");

	char temp_arrange;
	for (int i = 0; i < arr_arrange; i++) {
		for (int j = 0; j < arr_arrange - 1; j++) {
			if (arr1[j] > arr1[j + 1]) {
				temp_arrange = arr1[j];
				arr1[j] = arr1[j + 1];
				arr1[j + 1] = temp_arrange;
			}
		}
	}
	
	FILE* fp2 = fopen("hw3_output2.txt", "w");
	for (int i = 0; i < arr_arrange; i++) {
		printf("%c : ", arr1[i]);
		fprintf(fp2, "%c", arr1[i]);
		fprintf(fp2, "%s", " : ");
		Finder(final, arr1[i], result, -1, fp2);
		fprintf(fp2, "%s", "\n");
		printf("\n");
	}

	for (int i = 0; i < arr0_count; i++) {
		printf("%c", arr0[i]);
		fprintf(fp2, "%c", arr0[i]);
	}
	fclose(fp2);

	return 0;
}
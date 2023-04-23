/**
 * Splits an array to have floor(length*percent) elements
 * in the first chunk and the rest in the second chunk
 * @param {Array<any>} arr 
 * @param {number} percent 
 * @returns 
 */
export function splitArray(arr, percent) {
    const n = arr.length;
    const splitIndex = Math.floor(percent * n);
    const firstChunk = arr.slice(0, splitIndex);
    const secondChunk = arr.slice(splitIndex);
    return [firstChunk, secondChunk];
}

/**
 * This function separates a certain column from the rest of the 2D array
 * @param {Array<Array<any>>} dataset a 2D array
 * @param {int} labelColumn to be separated
 * @returns an array [X,Y] where X,Y are 2D arrays
 */
export function splitDataset(dataset, labelColumn) {
    const X = [];
    const Y = [];
    for (let i = 0; i < dataset.length; i++) {
        const row = dataset[i];
        const x = row.filter((element, index) => index !== labelColumn);
        const y = row[labelColumn];
        X.push(x);
        Y.push([y]);
    }
    return [X, Y];
}
/**
 * Converts a 2D array of strings that comes from a csv and converts every value to a float
 * It also shuffles the data
 * @param {Array<Array<string>>} dataString comes from reading a csv file
 * @returns {Array<Array<number>>}
 */
export function convertCsvDataset(dataString) {
    let allData = $.csv.toArrays(dataString)
    for (let i = 0; i < allData.length; i++)
        for (let j = 0; j < allData[i].length; j++)
            allData[i][j] = parseFloat(allData[i][j]);
    allData.sort(() => Math.random() - 0.5);
    return allData;
}

/**
 * 
 * @param {Array<Array<string>>} dataString comes from reading a csv file
 * @param {*} task the classification task from tasks.json
 * @returns {Array<tensor>} [dataX,dataY]
 */
export function readClassificationDataset(dataString, task) {
    let allData = convertCsvDataset(dataString)
    const [dataXArr, dataYArr] = splitDataset(allData, task.labelColumn);

    const dataX = tf.tensor(dataXArr)
    const dataY = tf.oneHot(tf.cast(tf.tensor(dataYArr), 'int32').squeeze(), task.classes)
    return [dataX, dataY]
}
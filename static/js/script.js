
window.onload = function () {
  createDropDownList('/api/tablelist');
}
// create float window
function createDropDownList(url) {
  fetch(url)
    .then(function (response) {
      return response.json();
    })
    .then(function (content) {

      console.log(content);

      let tableList = document.getElementById('tableList');

      content.forEach(element => {
        let li = document.createElement('li');
        let a = document.createElement('a');
        a.classList.add('dropdown-item');
        a.innerHTML = element.Value;
        // send table name to showTable function
        a.addEventListener('click', showTable.bind(null, element.Value));
        li.appendChild(a);
        tableList.appendChild(li);
      });
    });
}

// post query to backend and show result on resultText
function getSql() {

  let url = '/api/sql';
  let data = {
    question: document.getElementById('questionText').value,
    table_name: document.getElementById('tableName').innerHTML
  };

  fetch(url, {
    method: 'POST', // or 'PUT'
    body: JSON.stringify(data), // data can be `string` or {object}!
    headers: new Headers({
      'Content-Type': 'application/json'
    })
  }).then(res => res.json())
    .catch(error => console.error('Error:', error))
    .then(function (response) {
      let resultText = document.getElementById('resultText');
      resultText.innerHTML = response;
    });
}


// select and show table from database
function showTable(tableId) {

  // display table name
  let tableName = document.getElementById('tableName');
  tableName.innerHTML = tableId;

  let data = {table_name:tableId};
  let url = '/api/table';
  fetch(url, {
    method: 'POST', // or 'PUT'
    body: JSON.stringify(data), // data can be `string` or {object}!
    headers: new Headers({
      'Content-Type': 'application/json'
    })
  }).then(res => res.json())
  .catch(error => console.error('Error:', error))
  .then(function (response) {
    let table = createTable(response.columns, response.datas);
    let demoTable = document.getElementById('demoTable');
    demoTable.removeChild(demoTable.childNodes[0]); demoTable.appendChild(table);
  });
}


// create table
// refer from https://www.delftstack.com/zh-tw/howto/javascript/create-table-javascript/
function createTable(columns, datas) {
  let headers = [];
  let table = document.createElement('table');
  table.classList.add('table');
  table.classList.add('table-hover');

  let thead = document.createElement('thead');
  thead.classList.add('table-dark');

  let tbody = document.createElement('tbody');

  table.appendChild(thead);
  table.appendChild(tbody);

  // create table header
  tr = document.createElement('tr');
  columns.forEach(element => {
    let row = document.createElement('th');
    row.innerHTML = element.Name
    tr.appendChild(row);
    headers.push(element.Index);
  });
  thead.appendChild(tr)

  // create table body
  datas.forEach(element => {
    let row = document.createElement('tr');
    headers.forEach(key => {
      let rowData = document.createElement('td');
      rowData.classList.add("bar");
      rowData.innerHTML = element[key];
      row.appendChild(rowData);
    });
    tbody.appendChild(row);
  });
  return table;
}

// execute SQL command and show table
function runSql() {

  console.log('run sql...')

  fetch( '/api/runsql',{
    method: 'POST', // or 'PUT'
    body: JSON.stringify({sql:document.getElementById('resultText').value}), // data can be `string` or {object}!
    headers: new Headers({
      'Content-Type': 'application/json'
    })
  }).then(response=>response.json())
  .then(
    function (response) {
      console.log(response);
      // popout modal
      table = createTable(response.columns, response.datas);
      let modalTable = document.getElementById('modalTable');
      modalTable.removeChild(modalTable.childNodes[0]);
      modalTable.appendChild(table);
  });
};

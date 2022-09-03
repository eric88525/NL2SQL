
$(document).ready(function () {
  // create float window
  $("#floatWindow").kendoWindow({
    width: '1000px',
    height: 'auto',
    title: "SQL result",
    visible: false,
    modal: true,
    actions: [
      "Close"
    ],
  }).data("kendoWindow").center();

  createDropDownList('/api/tablelist');



  function createDropDownList(url) {
    content = []
    $.ajax({
      type: "get",
      url: url,
      dataType: "json",
      async: false,
      success: function (data) {
        content = data;
      }
    });

    let tableList = document.getElementById('tableList');

    tableList.removeChild(tableList.childNodes[0]);
    ul = document.createElement('ul');
    ul.classList.add('dropdown-menu');
    ul.classList.add('dropdown-menu-dark');
    tableList.appendChild(ul);

    content.forEach(element => {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.classList.add('dropdown-item');
      a.innerHTML = element.Value;
      // send table name to showTable function
      a.addEventListener('click', showTable.bind(null, element.Value));
      li.appendChild(a);
      ul.appendChild(li);
    });
  }




  // execute SQL command and show table

});


// post query to backend and show result on resultText
function getSql() {
  $.ajax({
    type: "post",
    url: '/api/sql',
    contentType: 'application/json;charset=UTF-8',
    data: JSON.stringify({
      question:  document.getElementById('questionText').value ,
      table_name: document.getElementById('tableName').innerHTML 
    }),
    async: false,
    success: function (data) {
      var resultText = document.getElementById('resultText');
      resultText.innerHTML = data;
    }
  });

}


// select and show table from database
function showTable(tableId) {

  // display table name
  let tableName = document.getElementById('tableName');
  tableName.innerHTML = tableId;

  var columns = []
  var datas = []

  $.ajax({
    type: "post",
    url: '/api/table',
    contentType: 'application/json;charset=UTF-8',
    data: JSON.stringify({
      table_name: tableId
    }),
    async: false,
    success: function (data) {
      columns = data.columns,
        datas = data.datas
    }
  });
  var table = createTable(columns, datas);

  let demoTable = document.getElementById('demoTable');
  demoTable.removeChild(demoTable.childNodes[0]);
  $('#demoTable').append(table);

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
    var row = document.createElement('th');
    row.innerHTML = element.Name

    tr.appendChild(row);
    headers.push(element.Index)
  });
  thead.appendChild(tr)

  // create table body
  datas.forEach(element => {
    var row = document.createElement('tr');
    headers.forEach(key => {
      var rowData = document.createElement('td');
      rowData.classList.add("bar");
      rowData.innerHTML = element[key]
      row.appendChild(rowData);
    });
    tbody.appendChild(row);
  });
  return table
}


function runSql() {
  console.log('run sql...')
  var columns = [];
  var datas = [];
  $.ajax({
    type: "post",
    url: '/api/runsql',
    contentType: 'application/json;charset=UTF-8',
    data: JSON.stringify({
      sql: $("#resultText").val()
    }),
    async: false,
    success: function (data) {
      columns = data.columns,
        datas = data.datas
    }
  });
  table = createTable(columns, datas);

  $("#floatDemoTable table").remove();
  $('#floatDemoTable').append(table);

  $("#floatWindow").data("kendoWindow").open();
}
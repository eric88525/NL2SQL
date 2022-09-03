
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

    // select table from database
    $(".tableItem").click(function () {
      // display table name
      $("#tableName").text(this.id);
      tableId = this.id;
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
      table = createTable(columns, datas);

      let demoTable = document.getElementById('demoTable');
      demoTable.removeChild(demoTable.childNodes[0]);
      $('#demoTable').append(table);

    })

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
      $("#tableList a").remove();
      content.forEach(element => {
        //console.log(element)
        var row = $("<a class='tableItem'   href='#' id='" + element.Value + "'></a>").addClass("dropdown-item").text(element.Text)
        $("#tableList").append(row)
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

    // create table object
    /*
    function createTable(columns, datas) {

      var headers = [];

      var thead = $("<thead></thead>").addClass('thead-dark ');
      var table = $('<table></table>').addClass('table table-striped');
      var body = $('<tbody></tbody>');

      tr = $('<tr></tr>');
      columns.forEach(element => {
        var row = $("<th></th>").text(element.Name);
        tr.append(row);
        headers.push(element.Index)
      });
      thead.append(tr)

      datas.forEach(element => {
        var row = $("<tr></tr>");
        headers.forEach(key => {
          var rowData = $('<td></td>').addClass('bar').text(element[key]);
          row.append(rowData);
        });
        body.append(row);
      });

      table.append(thead);
      table.append(body);
      return table
    }
    */

    // post query to backend and show result on resultText
    $("#transSql").click(function () {
      $.ajax({
        type: "post",
        url: '/api/sql',
        contentType: 'application/json;charset=UTF-8',
        data: JSON.stringify({
          question: $("#questionText").val(),
          table_name: $("#tableName").text()
        }),
        async: false,
        success: function (data) {
          $("#resultText").val(data)
        }
      });

    });
    // execute SQL command and show table
    $("#runSql").click(function () {
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
    });
  });

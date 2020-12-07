$(document).ready(function(){
    $("#tableNameList").kendoDropDownList({
        readonly: true,
        optionLabel: "請選擇",
        dataTextField: "Text",
        dataValueField: "Value",
        dataSource: {
          transport: {
            read: {
                url: "/api/tablelist", 
                type: "get",
                dataType: "json"
            }
          }
        },               
    });
    $("#search").click(function(){
        

    })

});
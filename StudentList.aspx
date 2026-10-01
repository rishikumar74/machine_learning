<%@ Page Title="Student Directory" Language="C#" MasterPageFile="~/Site1.Master" AutoEventWireup="true" CodeBehind="StudentList.aspx.cs" Inherits="WebApplication1.StudentList" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <h2>Enrolled Student Directory</h2>
    <p>Search and filter registered students currently active in the institution.</p>

    <div style="margin: 15px 0;">
        <asp:TextBox ID="txtSearch" runat="server" placeholder="Search by Name or Branch"></asp:TextBox>
        <asp:Button ID="btnFilter" runat="server" Text="Filter" OnClick="btnFilter_Click" />
        <asp:Button ID="btnReset" runat="server" Text="Show All" OnClick="btnReset_Click" />
    </div>

    <asp:GridView ID="gvStudents" runat="server" AutoGenerateColumns="True" CssClass="custom-grid">
    </asp:GridView>
</asp:Content>
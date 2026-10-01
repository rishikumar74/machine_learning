<%@ Page Title="Register Student" Language="C#" MasterPageFile="~/Site1.Master" AutoEventWireup="true" CodeBehind="StudentRegistration.aspx.cs" Inherits="WebApplication1.StudentRegistration" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <h2>Student Registration Form</h2>
    <p>Enter the details below to enroll a new student in the CVR database.</p>
    
    <table class="form-table">
        <tr>
            <td><label>Roll Number:</label></td>
            <td><asp:TextBox ID="txtRollNo" runat="server"></asp:TextBox></td>
        </tr>
        <tr>
            <td><label>Full Name:</label></td>
            <td><asp:TextBox ID="txtName" runat="server"></asp:TextBox></td>
        </tr>
        <tr>
            <td><label>Email ID:</label></td>
            <td><asp:TextBox ID="txtEmail" runat="server"></asp:TextBox></td>
        </tr>
        <tr>
            <td><label>Branch:</label></td>
            <td>
                <asp:DropDownList ID="ddlBranch" runat="server">
                    <asp:ListItem Text="Computer Science &amp; Engineering (CSE)" Value="CSE" />
                    <asp:ListItem Text="CSE (Artificial Intelligence &amp; ML)" Value="CSM" />
                    <asp:ListItem Text="CSE (Data Science)" Value="CSD" />
                    <asp:ListItem Text="Information Technology (IT)" Value="IT" />
                    <asp:ListItem Text="Electronics &amp; Communication (ECE)" Value="ECE" />
                </asp:DropDownList>
            </td>
        </tr>
        <tr>
            <td></td>
            <td>
                <asp:Button ID="btnSubmit" runat="server" Text="Register Student" OnClick="btnSubmit_Click" />
            </td>
        </tr>
    </table>
    
    <div style="margin-top: 15px;">
        <asp:Label ID="lblMessage" runat="server" Font-Bold="true"></asp:Label>
    </div>
</asp:Content>